import torch
from torch import nn
from einops import rearrange

import numpy as np
import math


class BlockDistanceConv(nn.Module):
    """
    A 1x1 convolution layer with weights based on spatial distances between blocks.
    """

    def __init__(
        self, num_patches_per_side=16, patch_group_size=16, transform="linear", local_thres=1.5, exp_sigma=3
    ):
        """
        Args:
            num_patches_per_side: Number of patches per side (e.g., 16 for 16x16 patches)
            patch_group_size: Number of patches in each block (default: 16)
            transform: Transform function to apply to distances ('linear', 'cos', 'exp', 'gaussian')
        """
        super().__init__()

        self.num_patches_per_side = num_patches_per_side
        self.patch_group_size = patch_group_size
        self.transform = transform
        self.local_thres = local_thres  # Threshold for local connections, can be adjusted
        self.exp_sigma = exp_sigma

        # Calculate number of blocks per side
        patches_per_block_side = int(np.sqrt(patch_group_size))
        assert patches_per_block_side > 0, "patch_group_size must be a perfect square > 0."
        self.blocks_per_side = (
            num_patches_per_side // patches_per_block_side
        )
        self.total_blocks = self.blocks_per_side**2

        # Create distance matrix
        distance_matrix = self._compute_block_distances()

        # Apply transformation
        weight_matrix = self._apply_transform(distance_matrix)

        # Create 1x1 conv layer
        self.conv = nn.Conv2d(
            in_channels=self.total_blocks,
            out_channels=self.total_blocks,
            kernel_size=1,
            bias=False,
        )

        # Set the weights as fixed (no gradient)
        with torch.no_grad():
            self.conv.weight.data = weight_matrix.unsqueeze(-1).unsqueeze(-1)


    def _compute_block_distances(self):
        """Compute Euclidean distances between all block centers."""
        # Get block center coordinates
        block_centers = []
        for i in range(self.blocks_per_side):
            for j in range(self.blocks_per_side):
                # Center of block in grid coordinates
                center_x = i + 0.5
                center_y = j + 0.5
                block_centers.append([center_x, center_y])

        block_centers = torch.tensor(block_centers, dtype=torch.float32)

        # Compute pairwise distances
        distance_matrix = torch.zeros(self.total_blocks, self.total_blocks)

        for i in range(self.total_blocks):
            for j in range(self.total_blocks):
                dist = torch.norm(block_centers[i] - block_centers[j], p=2)
                distance_matrix[i, j] = dist

        return distance_matrix

    def _apply_transform(self, distance_matrix):
        """Apply transformation function to distance matrix."""
        if self.transform == "linear":
            max_dist = distance_matrix.max()
            mat = 1.0 - (distance_matrix / max_dist)
            return mat / mat.sum(dim=0, keepdim=True)

        elif self.transform == "cos":
            max_dist = distance_matrix.max()
            normalized_dist = distance_matrix / max_dist * math.pi / 4
            mat = torch.cos(normalized_dist)
            return mat / mat.sum(dim=0, keepdim=True)

        elif self.transform == "exp":
            mat = torch.exp(-distance_matrix / self.exp_sigma)
            return mat / mat.sum(dim=0, keepdim=True)

        elif self.transform == "gaussian":
            sigma = distance_matrix.max() / 3
            return torch.exp(-(distance_matrix**2) / (2 * sigma**2))
        
        elif self.transform == "local":
            thres = getattr(self, "local_thres", 1.5)
            mat = (distance_matrix <= thres).float()
            mat = mat / mat.sum(dim=0, keepdim=True)
            return mat

        else:
            raise ValueError(f"Unknown transform: {self.transform}")

    def forward(self, x):
        return self.conv(x.to(dtype=self.conv.weight.dtype))

    def get_weight_matrix(self):
        return self.conv.weight.data.squeeze(-1).squeeze(-1)


class MHLA_ForSana(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=None,
        dropout=0.1,
        fixed_weight_value=None,
        qk_norm=False,
        transform="cos",
        **kwargs,
    ):
        super(MHLA_ForSana, self).__init__()

        if dim_head is None:
            dim_head = dim // heads
        inner_dim = dim_head * heads
        self.num_heads = heads
        self.head_dim = dim_head
        self.scale = dim_head**-0.5
        self.dim = dim

        self.norm = nn.LayerNorm(dim)
        is_bias = kwargs.get("qkv_bias", False)
        self.qkv = nn.Linear(dim, inner_dim * 3, bias=is_bias)

        self.q_norm = nn.RMSNorm(dim) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(dim) if qk_norm else nn.Identity()
        
        self.num_pieces = kwargs.get("window_size", 16)
        assert int(self.num_pieces**0.5)**2 == self.num_pieces, "window_size must be a perfect square."
        self.hb = self.wb = int(self.num_pieces**0.5)

        local_thres = kwargs.get("local_thres", 1.5)
        exp_sigma = kwargs.get("exp_sigma", 3)

        self.lepe = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        
        self.piece_attn = BlockDistanceConv(
            num_patches_per_side=self.hb,
            patch_group_size=1,
            transform=transform,
            local_thres=local_thres,
            exp_sigma=exp_sigma,
        )

        self.eps = kwargs.get("eps", 1e-6)

        # self.proj = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.proj = nn.Linear(inner_dim, dim, bias=True)

        if fixed_weight_value is not None:
            self._init_weights_with_fixed_value(fixed_weight_value)

    def _init_weights_with_fixed_value(self, value):
        for name, param in self.named_parameters():
            if "weight" in name and param.requires_grad:
                nn.init.constant_(param, value)
            elif "bias" in name and param is not None:
                nn.init.zeros_(param)

        nn.init.constant_(self.qkv.weight, value)

        for module in self.proj:
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, value)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def init_to_value(model, value=1.0):
        for name, param in model.named_parameters():
            if "weight" in name and param.requires_grad:
                nn.init.constant_(param, value)
            elif "bias" in name and param is not None:
                nn.init.zeros_(param)
        return model

    # @torch.compile
    def _process_qkv_impl(self, q, k, v, B, N, H, D):
        q = self.q_norm(q)
        k = self.k_norm(k)

        k = torch.relu(k) + self.eps
        q = torch.relu(q) + self.eps

        q, k, v = map(
            lambda t: rearrange(
                t, "b n w (h d) -> (b h) n w d", h=H, d=D
            ),
            (q, k, v)
        )

        k = k.transpose(-2, -1)
        return q, k, v

    # @torch.compile
    def _mlp_lepe(self, x, p1, p2):
        qkv = self.qkv(x)
        
        lepe = self.lepe(rearrange(qkv[..., self.dim * 2:], 'b h w d -> b d h w'))
        lepe = rearrange(lepe, 'b d (h p1) (w p2) -> b (h w) (p1 p2) d', h=self.hb, w=self.wb, p1=p1, p2=p2)
        qkv = rearrange(qkv, 'b (hb p1) (wb p2) c -> b (hb wb) (p1 p2) c', hb=self.hb, wb=self.wb, p1=p1, p2=p2)
        q, k, v = qkv.chunk(3, dim=-1)        
        
        return q, k, v,lepe


    def forward(self, x: torch.Tensor, HW: tuple, image_rotary_emb: torch.Tensor = None) -> torch.Tensor:
        height, width = HW
        x = self.norm(x)
        B, N, C = x.shape

        x = rearrange(x, 'b (h w) c -> b h w c', h=height, w=width)
        
        p1 = height // self.hb
        p2 = width // self.wb

        # x = rearrange(
        #     x,
        #     'b (hb p1) (wb p2) c -> b (hb wb) (p1 p2) c',
        #     hb=self.hb, wb=self.wb, p1=p1, p2=p2
        # )
        
        # B, num_pieces, piece_size_flat, C = x.shape
        # H = self.num_heads
        # D = self.head_dim
        # q, k, v = self.qkv(x).chunk(3, dim=-1)

        q, k, v, lepe = self._mlp_lepe(x, p1, p2)

        q, k, v = q.float(), k.float(), v.float()
        q, k, v = self._process_qkv_impl(q, k, v, B, self.num_pieces, self.num_heads, self.head_dim)

        kv = torch.matmul(k, v)
        # piece_attn conv runs in its param dtype (e.g. bf16); q/k/v stay fp32 above for stability
        kv = self.piece_attn(kv).to(dtype=q.dtype)

        k_sum = k.sum(dim=-1, keepdim=True)
        normalizer = self.piece_attn(torch.matmul(q, k_sum)).to(dtype=q.dtype) + self.eps

        out = torch.matmul(q, kv) / normalizer
        out = rearrange(out, "(b h) n w d -> b n w (h d)", b=B, h=self.num_heads)
        out = out + lepe.to(dtype=out.dtype)

        out = rearrange(
            out,
            'b (hb wb) (p1 p2) c -> b (hb p1) (wb p2) c',
            hb=self.hb, wb=self.wb, p1=p1, p2=p2
        )
        out = rearrange(out, 'b h w c -> b (h w) c')

        w_dtype = self.proj.weight.dtype
        return self.proj(out.to(dtype=w_dtype))
