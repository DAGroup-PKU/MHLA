from einops import rearrange
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer import checkpoint_seq
import torch
import torch.nn.functional as F



class MHLA_ViT(VisionTransformer):
    def __init__(self, *args, piece_size=4, padding=True, flatten=True, is_conv=False, **kwargs):
        super(MHLA_ViT, self).__init__(*args, **kwargs)
        self.piece_size = piece_size
        
        self.flatten = flatten
        self.padding = padding
        self.is_conv = is_conv
        self.patch_size = kwargs.get('patch_size', 16)  # 默认16
        
        

    def rearrange_patches(self, x: torch.Tensor) -> torch.Tensor:
        # 不考虑 cls token，x: [B, N, C]
        B, N, C = x.shape
        piece = self.piece_size
        H = W = int(N ** 0.5)
        assert H * W == N, "Patch数量不是正方形，暂不支持"
        assert H % piece == 0 and W % piece == 0, "H/W 必须能被 piece_size 整除"

        # 先还原为2D，再以piece为单位重排
        x = rearrange(
            x, 
            'b (h w) c -> b h w c', 
            h=H, w=W
        )
        if not self.is_conv:
            x = rearrange(
                x,
                'b (hb p1) (wb p2) c -> b (hb wb) (p1 p2) c',
                hb=H // piece, wb=W // piece, p1=piece, p2=piece
            )

        if self.flatten:
            x = rearrange(
                x,
                'b n s c -> b (n s) c'
            )
        return x

    def pad_to_16x16_patches(self, img):
        # img: [B, C, H, W]
        B, C, H, W = img.shape
        target_size = self.patch_size * 16  # 256
        pad_h = target_size - H
        pad_w = target_size - W
        # pad顺序: (left, right, top, bottom)
        img = F.pad(img, (pad_w // 2, pad_w // 2, pad_h // 2, pad_h // 2), value=0)
        return img
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding:
            x = self.pad_to_16x16_patches(x)
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.rearrange_patches(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        if not self.flatten:
            x = rearrange(x, 'b n w d -> b (n w) d')
        x = self.norm(x)
        return x
    
    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
    
        x = self.pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)


def strict_piecewise_rearrange(x, piece):
    # x: [B, N, C]
    B, N, C = x.shape
    H = int(N ** 0.5)
    W = N // H
    assert H * W == N
    assert H % piece == 0 and W % piece == 0
    x2d = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
    x_out = rearrange(
        x2d,
        'b (nh ph) (nw pw) c -> b (nh nw ph pw) c',
        nh=H // piece, ph=piece, nw=W // piece, pw=piece
    )
    return x_out



