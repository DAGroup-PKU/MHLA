import torch
from torch import nn
from torch.cuda import amp
from einops import rearrange
import math



class BlockDistanceConv3D(nn.Module):
    """
    A 1x1 convolution layer with weights based on spatial-temporal distances between blocks.
    Adapted for video models with 3D token structure.
    """

    def __init__(
        self,
        blocks_layout=(4, 4, 4),
        transform="linear",
        local_thres=1.5,
        exp_sigma=3,
    ):
        """
        Args:
            num_frames: Number of frames in the video
            num_patches_per_side: Number of patches per side (e.g., 16 for 16x16 patches)
            patch_group_size: Number of patches in each block (default: 16)
            transform: Transform function to apply to distances ('linear', 'cos', 'exp', 'gaussian')
        """
        super().__init__()

        self.blocks_layout = blocks_layout
        self.transform = transform
        self.local_thres = local_thres
        self.exp_sigma = exp_sigma

        # Calculate number of blocks per dimension
      
        self.total_blocks = blocks_layout[0] * blocks_layout[1] * blocks_layout[2]

        # Create distance matrix
        distance_matrix = self._compute_block_distances_3d()

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
            # For 1x1 conv: (total_blocks, total_blocks, 1, 1)
            self.conv.weight.data = weight_matrix.unsqueeze(-1).unsqueeze(-1)

    def _compute_block_distances_3d(self):
        """Compute Euclidean distances between all 3D block centers."""
        # Get 3D block center coordinates
        block_centers = []
        for f in range(self.blocks_layout[0]):
            for i in range(self.blocks_layout[1]):
                for j in range(self.blocks_layout[2]):
                    # Center of block in 3D grid coordinates (frame, x, y)
                    center_f = f + 0.5
                    center_x = i + 0.5
                    center_y = j + 0.5
                    block_centers.append([center_f, center_x, center_y])

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
            mat = torch.exp(-distance_matrix / self.exp_sigma)
            return mat / mat.sum(dim=0, keepdim=True)

        elif self.transform == "gaussian":
            # Gaussian kernel
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
        return self.conv(x)

    def get_weight_matrix(self):
        """Return the weight matrix for inspection."""
        return self.conv.weight.data.squeeze(-1).squeeze(-1)

@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()

class MHLA_Video_Uni(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        dim_head=None,
        dropout=0.1,
        fixed_weight_value=None,
        qk_norm=True,
        block_layout=(3, 5, 10),
        transform="linear",
        qkv_bias=False,
        eps=1e-6,
        is_gated=False,
        is_lepe=False,
        **kwargs,
    ):
        from diffusion.model.wan.model import WanRMSNorm
        """
        Args:
            dim: Dimension of the input features
            heads: Number of heads
            dim_head: Dimension of each head
            dropout: Dropout rate
            fixed_weight_value: Fixed weight value
            qk_norm: Whether to use RMSNorm for q and k
            block_size: Size of the blocks
            input_shape: Shape of the input
            transform: Transform function to apply to the distances
        """
        super(MHLA_Video_Uni, self).__init__()

        dim_head = dim // num_heads
        dim_head * num_heads

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim_head
        # self.block_size = block_size
        # self.input_shape = input_shape
        # self.norm = nn.LayerNorm(dim)
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.g = nn.Linear(dim, dim) if is_gated else None
        self.g_fn = nn.SiLU() if is_gated else None
        self.g_norm = WanRMSNorm(dim_head, eps=eps)

        self.is_gated = is_gated
        self.is_lepe = is_lepe

        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.out_norm = kwargs.get("out_rmsnorm", False)
        # self.out_rmsnorm = WanRMSNorm(dim, eps=eps) if self.out_norm else nn.Identity()
        self.normalize_out = kwargs.get("normalize_out", True)

        # Video-specific parameters
        # self.embed_len = input_shape[0] * input_shape[1] * input_shape[2]
        self.blocks_layout = block_layout
        self.num_blocks = self.blocks_layout[0] * self.blocks_layout[1] * self.blocks_layout[2]
        
        
        self.block_attn = BlockDistanceConv3D(
            blocks_layout=self.blocks_layout,
            transform=transform,
        )

        self.lepe = nn.Conv3d(dim, 
                            dim, 
                            kernel_size=(3, 3, 3), 
                            stride=1,
                            padding=(1, 1, 1),
                            groups=dim) if is_lepe else None

        self.eps = eps
        self.o = nn.Linear(dim, dim)
        self.rope_after = kwargs.get("rope_after", False)
        self.power = kwargs.get("power", 1.0)
        self.without_rope = kwargs.get("without_rope", False)

        if fixed_weight_value is not None:
            self._init_weights_with_fixed_value(fixed_weight_value)

    def _init_weights_with_fixed_value(self, value):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.constant_(param, value)
            elif "bias" in name and param is not None:
                nn.init.zeros_(param)

        nn.init.constant_(self.q.weight, value)
        nn.init.constant_(self.k.weight, value)
        nn.init.constant_(self.v.weight, value)
        nn.init.constant_(self.o.weight, value)



    @staticmethod
    def init_to_value(model, value=1.0):
        for name, param in model.named_parameters():
            if "weight" in name:
                nn.init.constant_(param, value)
            elif "bias" in name and param is not None:
                nn.init.zeros_(param)
        return model

    # @torch.compile
    def _process_qkv_impl(self, q, k, v, B, N, H, D):
        q = self.norm_q(q)  # [B, H, N, D]
        k = self.norm_k(k)  # [B, H, N, D]

        k = torch.relu(k) + self.eps
        q = torch.relu(q) + self.eps

        # k = k.transpose(-2, -1)

        return q, k, v

    # @torch.compile
    def _qkv_fn(self, x, F, H, W):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        if self.is_lepe:
            lepe = self.lepe((rearrange(v, 'b (f h w) c -> b c f h w', f=F, h=H, w=W)))
            lepe = rearrange(lepe, 'b c f h w -> b (f h w) c')
        else:
            lepe = None
        return q, k, v, lepe
        # return q, k, v

    # @torch.compile
    def forward(self, x: torch.Tensor, seq_lens, grid_sizes, freqs) -> torch.Tensor:
        # Input shape: [B, F, H, W, C]
        B, N, C = x.shape
        
        # B, F, H, W, C = x.shape
        # N = F * H * W  # Total number of tokens
        F, H, W = grid_sizes[0, 0], grid_sizes[0, 1], grid_sizes[0, 2]
        
        block_size = (F // self.blocks_layout[0], H // self.blocks_layout[1], W // self.blocks_layout[2])
        
        # x = self.norm(x)
        q, k, v, lepe =  self._qkv_fn(x, F, H, W)
        dtype = q.dtype
        # x = rearrange(x, "b (f h w) c -> b f h w c", f=F, h=H, w=W)
        # qkv, lepe = self._mlp_lepe(x)

        q, k, v = q.float(), k.float(), v.float()
        q, k, v = self._process_qkv_impl(q, k, v, B, N, self.num_heads, self.head_dim)
        # print(q.shape)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b n h d", h=self.num_heads), (q, k, v)
        )
        q_rope, k_rope = rope_apply(q, grid_sizes, freqs), rope_apply(k, grid_sizes, freqs)
        
        # Rearrange to pieces for attention
        q, k, v, q_rope, k_rope = rearrange(
            torch.cat([q, k, v, q_rope, k_rope], dim=-1),
            "b (fb p1 hb p2 wb p3) h c -> (b h) (fb hb wb) (p1 p2 p3) c",
            fb=self.blocks_layout[0],
            hb=self.blocks_layout[1],
            wb=self.blocks_layout[2],
            p1=block_size[0],
            p2=block_size[1],
            p3=block_size[2],
        ).chunk(5, dim=-1) # [B, num_blocks, block_size, D]
        
        k_rope = k_rope.transpose(-2, -1)
        k = k.transpose(-2, -1)

        kv = torch.matmul(k_rope, v)  # [B*H, num_blocks, D, D]
        kv = self.block_attn(kv)  # [B*H, num_blocks, D, D]

        if self.normalize_out:
            k_sum = k.sum(dim=-1, keepdim=True)  # [B*H, num_blocks, D, 1]
            normalizer = (
                self.block_attn(torch.matmul(q, k_sum)) + self.eps
            )  # [B*H, num_blocks, block_size, 1]
            out = torch.matmul(q_rope, kv) / normalizer  # [B*H, num_blocks, block_size, D]
        else:
            out = torch.matmul(q_rope, kv)  # [B*H, num_blocks, block_size, D]

        out = rearrange(out, "(b h) n w d -> b n w (h d)", b=B, h=self.num_heads)

        out = rearrange(
            out,
            "b (fb hb wb) (p1 p2 p3) c -> b (fb p1 hb p2 wb p3) c",
            fb=self.blocks_layout[0],
            hb=self.blocks_layout[1],
            wb=self.blocks_layout[2],
            p1=block_size[0],
            p2=block_size[1],
            p3=block_size[2],
        )

        out = out.to(dtype)
        if self.is_gated:
            g = self.g(x)
            g = self.g_fn(g)
            out = rearrange(self.g_norm(rearrange(out, "b n (h d) -> b n h d", h=self.num_heads)), "b n h d -> b n (h d)") * g
        else:
            out = rearrange(self.g_norm(rearrange(out, "b n (h d) -> b n h d", h=self.num_heads)), "b n h d -> b n (h d)")
        if self.is_lepe:
            out = out + lepe

        return self.o(out)