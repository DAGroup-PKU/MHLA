import torch
from torch import nn

from einops import rearrange

class RMSNorm(nn.Module):
    def __init__(self, dim: int, scale_factor=1.0, eps: float = 1e-6):
        """
            Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim) * scale_factor)

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        return (self.weight * self._norm(x.float())).type_as(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, dropout=0., **kwargs):
        super().__init__()
        if dim_head is None:
            dim_head = dim // heads
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.q_norm = RMSNorm(dim, scale_factor=1.0, eps=1e-6)
        self.k_norm = RMSNorm(dim, scale_factor=1.0, eps=1e-6)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), [q, k, v])

        q, k = map(lambda t: torch.nn.functional.relu(t), (q, k))

        k = k.transpose(-1, -2)
        kv = torch.matmul(k, v)
        out = torch.matmul(q, kv)

        k_sum = k.sum(dim=-1, keepdim=True)
        normalizer = torch.matmul(q, k_sum) + 1e-6  #

        out = out / normalizer  # [B*H, num_pieces, window_size, D]


        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
