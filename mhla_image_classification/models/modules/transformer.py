from torch import nn


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., attn_type=None, ffn_type=None, attn_kwargs=None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.attn_type = attn_type
        self.ffn_type = ffn_type
        
        # 确保attn_kwargs是字典
        attn_kwargs = attn_kwargs or {}
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                self.attn_type(
                    dim, 
                    heads=heads, 
                    dim_head=dim_head, 
                    dropout=dropout,
                    **attn_kwargs  # 传递额外的参数给注意力层
                ),
                self.ffn_type(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)