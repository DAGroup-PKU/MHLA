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
        patches_per_block_side = int(np.sqrt(patch_group_size))  # 4 for 16 patches
        self.blocks_per_side = (
            num_patches_per_side // patches_per_block_side
        )  # 4 for 16x16 patches
        self.total_blocks = self.blocks_per_side**2  # 16 blocks

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
            # Weight shape for Conv2d: (out_channels, in_channels, kernel_h, kernel_w)
            # For 1x1 conv: (16, 16, 1, 1)
            self.conv.weight.data = weight_matrix.unsqueeze(-1).unsqueeze(-1)

        # Freeze the weights
        # self.conv.weight.requires_grad = False


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
        # distance_matrix[i, j] = distance from block i to block j
        distance_matrix = torch.zeros(self.total_blocks, self.total_blocks)

        for i in range(self.total_blocks):
            for j in range(self.total_blocks):
                dist = torch.norm(block_centers[i] - block_centers[j], p=2)
                distance_matrix[i, j] = dist

        return distance_matrix

    def _apply_transform(self, distance_matrix):
        """Apply transformation function to distance matrix."""
        if self.transform == "linear":
            # Normalize to [0, 1] and invert (closer blocks have higher weights)
            max_dist = distance_matrix.max()
            mat = 1.0 - (distance_matrix / max_dist)
            return mat / mat.sum(dim=0, keepdim=True)

        elif self.transform == "cos":
            # Cosine transformation
            max_dist = distance_matrix.max()
            normalized_dist = distance_matrix / max_dist * math.pi / 4
            mat = torch.cos(normalized_dist)
            return mat / mat.sum(dim=0, keepdim=True)

        elif self.transform == "exp":
            # Exponential decay
            sigma = distance_matrix.max() / 3  # Adjust decay rate
            mat = torch.exp(-distance_matrix / self.exp_sigma)
            return mat / mat.sum(dim=0, keepdim=True)

        elif self.transform == "gaussian":
            # Gaussian kernel
            sigma = distance_matrix.max() / 3
            return torch.exp(-(distance_matrix**2) / (2 * sigma**2))
        
        elif self.transform == "local":
            thres = getattr(
                self, "local_thres", 1.5
            )  # 可通过 self.local_thres 控制阈值
            mat = (distance_matrix <= thres).float()
            mat = mat / mat.sum(dim=0, keepdim=True)
            return mat

        else:
            raise ValueError(f"Unknown transform: {self.transform}")

    def forward(self, x):
        """
        Forward pass through the distance-based convolution.

        Args:
            x: Input tensor of shape (B, 16, H, W) where 16 is the number of blocks

        Returns:
            Output tensor of shape (B, 16, H, W)
        """
        return self.conv(x)

    def get_weight_matrix(self):
        """Return the weight matrix for inspection."""
        return self.conv.weight.data.squeeze(-1).squeeze(-1)


class MHLA_Normed_Torch(nn.Module):
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
        super(MHLA_Normed_Torch, self).__init__()

        if dim_head is None:
            dim_head = dim // heads
        inner_dim = dim_head * heads
        self.num_heads = heads
        self.head_dim = dim_head
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        is_bias = kwargs["qkv_bias"] if "qkv_bias" in kwargs else False
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=is_bias)

        self.q_norm = nn.RMSNorm(dim) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(dim) if qk_norm else nn.Identity()
        
        self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)

        self.window_size = kwargs["window_size"] if "window_size" in kwargs else 49
        self.window_len = int(self.window_size ** 0.5)
        self.embed_len = kwargs["embed_len"] if "embed_len" in kwargs else 196
        self.num_pieces = self.embed_len // self.window_size
        self.pieces_len = int(self.num_pieces**0.5)
        # self.piece_attn = nn.Conv2d(in_channels=self.num_pieces, out_channels=self.num_pieces, kernel_size=1)
        local_thres = kwargs.get("local_thres", 1.5)
        exp_sigma = kwargs.get("exp_sigma", 3)
        
        # 正确使用 torch.compile 编译子模块，避免 .compile() 返回 None 覆盖模块本身
        self.piece_attn = BlockDistanceConv(
            num_patches_per_side=int(self.embed_len**0.5),
            patch_group_size=self.window_size,
            transform=transform,
            local_thres=local_thres,
            exp_sigma=exp_sigma,
        )
        # try:
        #     self.piece_attn = torch.compile(self.piece_attn)
        # except Exception:
        #     # 如果环境不支持 compile，则回退为未编译模块
        #     print("❌ Piece Attention编译失败")
        #     pass

        self.eps = kwargs.get("eps", 1e-6)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        # # 如果指定了固定权重值，则初始化所有权重为该值
        if fixed_weight_value is not None:
            self._init_weights_with_fixed_value(fixed_weight_value)
        

        # print("✅ Piece Attention已编译")
        # print("✅ QKV处理和reshape已编译")


    def _init_weights_with_fixed_value(self, value):
        """将模型中的所有权重初始化为固定值"""
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.constant_(param, value)
            elif "bias" in name and param is not None:
                nn.init.zeros_(param)

        # 特别处理一些层
        nn.init.constant_(self.to_qkv.weight, value)

        # 如果to_out是Sequential，需要单独处理其中的Linear层
        for module in self.to_out:
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, value)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def init_to_value(model, value=1.0):
        """静态方法，用于将现有模型的权重初始化为固定值"""
        for name, param in model.named_parameters():
            if "weight" in name:
                nn.init.constant_(param, value)
            elif "bias" in name and param is not None:
                nn.init.zeros_(param)
        return model

    @torch.compile
    def _process_qkv_impl(self, q, k, v, B, N, H, D):
        
        q = self.q_norm(q)  # [B, H, N, D]
        k = self.k_norm(k)  # [B, H, N, D]

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

    @torch.compile
    def _mlp_lepe(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        lepe = self.lepe(rearrange(v, 'b (h w) (p1 p2) d -> b d (h p1) (w p2)', h=self.pieces_len, w=self.pieces_len, p1=self.window_len, p2=self.window_len))
        lepe = rearrange(lepe, 'b d (h p1) (w p2) -> b (h w) (p1 p2) d', h=self.pieces_len, w=self.pieces_len, p1=self.window_len, p2=self.window_len)
        return q, k, v,lepe


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        B, N, W, C = x.shape
        H = self.num_heads
        D = self.head_dim
        

        q, k, v, lepe = self._mlp_lepe(x)
        
        q, k, v = self._process_qkv_impl(q, k, v, B, N, H, D)

        kv = torch.matmul(k, v)  # [B*H, num_pieces, D, D]
        kv = self.piece_attn(kv)  # [B*H, num_pieces, D, D]
        # 统计kv结果里的activation的min和max，并考虑有没有nan

        k_sum = k.sum(dim=-1, keepdim=True) #[B*H, num_pieces, D, 1]
        normalizer = self.piece_attn(torch.matmul(q, k_sum)) + self.eps # [B*H, num_pieces, window_size, 1]

        out = torch.matmul(q, kv) / normalizer  # [B*H, num_pieces, window_size, D]
        # out = torch.matmul(q, kv) * self.scale
        # out = rearrange(out, "b n w d -> b (n w) d")
        out = rearrange(out, "(b h) n w d -> b n w (h d)", b=B, h=self.num_heads)
        # out = out * self.scale
        out = out + lepe

        return self.to_out(out)
