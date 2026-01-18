import torch
from torch import nn as nn
import torch.nn.functional as F

from timm.layers.format import Format, nchw_to
from timm.layers.trace_utils import _assert
from timm.layers.patch_embed import PatchEmbed
from einops import rearrange


class PiecewisePatchEmbed(PatchEmbed):
    """2D Image to Patch Embedding"""

    output_fmt: Format
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
        self,
        *args,
        piece_size: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.piece_size = piece_size

    @staticmethod
    def reverse_rearrange_patches(
        x: torch.Tensor, H: int, W: int, piece_size=4
    ) -> torch.Tensor:
        piece = piece_size

        hb = H // piece
        wb = W // piece

        x = rearrange(x, "b (n s) c -> b n s c", n=hb * wb, s=piece * piece)
        x = rearrange(
            x,
            "b (hb wb) (p1 p2) c -> b (hb p1) (wb p2) c",
            hb=hb,
            wb=wb,
            p1=piece,
            p2=piece,
        )
        x = rearrange(x, "b h w c -> b (h w) c", h=H, w=W)
        return x

    @staticmethod
    def rearrange_patches(x: torch.Tensor, piece_size=4) -> torch.Tensor:
        B, N, C = x.shape
        piece = piece_size
        H = W = int(N**0.5)

        x = rearrange(x, "b (h w) c -> b h w c", h=H, w=W)
        x = rearrange(
            x,
            "b (hb p1) (wb p2) c -> b (hb wb) (p1 p2) c",
            hb=H // piece,
            wb=W // piece,
            p1=piece,
            p2=piece,
        )
        x = rearrange(x, "b n s c -> b (n s) c")
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        piece = self.piece_size

        if self.img_size is not None:
            if self.strict_img_size:
                _assert(
                    H == self.img_size[0],
                    f"Input height ({H}) doesn't match model ({self.img_size[0]}).",
                )
                _assert(
                    W == self.img_size[1],
                    f"Input width ({W}) doesn't match model ({self.img_size[1]}).",
                )
            elif not self.dynamic_img_pad:
                _assert(
                    H % self.patch_size[0] == 0,
                    f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]}).",
                )
                _assert(
                    W % self.patch_size[1] == 0,
                    f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]}).",
                )
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))
        H = (H + pad_h) // self.patch_size[0]
        W = (W + pad_w) // self.patch_size[1]
        x = self.proj(x)  # NCHW
        if self.flatten:
            # x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
            x = rearrange(
                x,
                "b c (hb p1) (wb p2) -> b (hb wb) (p1 p2) c",
                hb=H // piece,
                wb=W // piece,
                p1=piece,
                p2=piece,
            )
            x = rearrange(x, "b n s c -> b (n s) c")

        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x
