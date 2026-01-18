import warnings
from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.drop import DropPath
from timm.layers.mlp import Mlp
from timm.models.vision_transformer import Block, LayerScale
from torch import Tensor


class FlashAttention(nn.Module):
    """
    Flash Attention实现，支持多种后端：
    1. flash-attn库 (默认推荐)
    2. PyTorch原生的scaled_dot_product_attention
    3. 标准注意力实现 (回退选项)

    专为视觉任务设计，不包含因果掩码。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        backend: str = "flash_attn",  # 默认使用flash_attn
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, (
            f"dim {dim} should be divisible by num_heads {num_heads}"
        )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.dtype = dtype or torch.float32

        # 检测并设置后端
        self.backend = self._detect_backend(backend)

        # QKV投影层
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Q和K的归一化层（如果启用）
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        # 注意力dropout
        self.attn_drop = nn.Dropout(attn_drop)

        # 输出投影层
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # 性能统计
        self.performance_stats = {
            "backend_used": self.backend,
            "forward_calls": 0,
            "total_time": 0.0,
        }

        print(f"FlashAttention初始化完成，使用后端: {self.backend}")

    def _detect_backend(self, requested_backend: str) -> str:
        """检测并选择最优的注意力后端"""
        if requested_backend == "standard":
            return "standard"

        # 检查flash-attn库
        flash_attn_available = False
        try:
            import flash_attn

            flash_attn_available = True
        except ImportError:
            flash_attn_available = False

        # 检查PyTorch的scaled_dot_product_attention
        pytorch_available = False
        try:
            # 测试是否支持scaled_dot_product_attention
            test_q = torch.randn(1, 1, 4, 4)
            test_k = torch.randn(1, 1, 4, 4)
            test_v = torch.randn(1, 1, 4, 4)
            _ = F.scaled_dot_product_attention(test_q, test_k, test_v)
            pytorch_available = True
        except (AttributeError, RuntimeError):
            pytorch_available = False

        # 根据请求和可用性选择后端
        if requested_backend == "auto":
            if flash_attn_available:
                return "flash_attn"
            elif pytorch_available:
                return "pytorch"
            else:
                return "standard"
        elif requested_backend == "flash_attn":
            if flash_attn_available:
                return "flash_attn"
            else:
                warnings.warn("flash-attn库不可用，回退到标准实现")
                return "standard"
        elif requested_backend == "pytorch":
            if pytorch_available:
                return "pytorch"
            else:
                warnings.warn(
                    "PyTorch scaled_dot_product_attention不可用，回退到标准实现"
                )
                return "standard"
        else:
            return "standard"

    def _pytorch_attention(
        self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Optional[Tensor] = None
    ) -> Tensor:
        """使用PyTorch原生的scaled_dot_product_attention"""
        # 使用PyTorch的Flash Attention实现
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,  # 图像不需要因果掩码
            scale=self.scale,
        )

        return out

    def _flash_attn_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """使用 flash-attn 0.2.7 实现"""
        try:
            from flash_attn.flash_attn_interface import flash_attn_unpadded_func

            # 输入: (batch, num_heads, seq_len, head_dim)
            batch, num_heads, seq_len, head_dim = q.shape

            # flash-attn 0.2.7 期望输入: (total, nheads, headdim)
            # 其中 total=batch*seq_len（flat 展开）
            # cu_seqlens 形如: [0, seq_len, 2*seq_len, ...]
            q_ = (
                q.permute(0, 2, 1, 3)
                .reshape(batch * seq_len, num_heads, head_dim)
                .contiguous()
            )
            k_ = (
                k.permute(0, 2, 1, 3)
                .reshape(batch * seq_len, num_heads, head_dim)
                .contiguous()
            )
            v_ = (
                v.permute(0, 2, 1, 3)
                .reshape(batch * seq_len, num_heads, head_dim)
                .contiguous()
            )

            cu_seqlens = torch.arange(
                0,
                (batch + 1) * seq_len,
                step=seq_len,
                dtype=torch.int32,
                device=q.device,
            )
            max_seqlen = seq_len

            if attn_mask is not None:
                warnings.warn("flash-attn 0.2.x 暂不支持显式 mask，将回退到标准实现")
                return self._standard_attention(q, k, v, attn_mask)

            # dropout_p 只在训练时生效
            dropout_p = self.attn_drop.p if self.training else 0.0

            out = flash_attn_unpadded_func(
                q_,
                k_,
                v_,
                cu_seqlens,
                max_seqlen,
                False,  # causal=False
                softmax_scale=getattr(
                    self, "scale", None
                ),  # 有的实现是 None，有的有 scale
            )

            # 输出: (total, nheads, headdim)
            out = (
                out.reshape(batch, seq_len, num_heads, head_dim)
                .permute(0, 2, 1, 3)
                .contiguous()
            )  # (batch, num_heads, seq_len, head_dim)
            return out

        except ImportError:
            warnings.warn("flash-attn库不可用，回退到标准实现")
            return self._standard_attention(q, k, v, attn_mask)

    def _standard_attention(
        self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Optional[Tensor] = None
    ) -> Tensor:
        """标准的注意力实现"""
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 应用掩码（如果提供）
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        # 计算输出
        out = torch.matmul(attn_probs, v)

        return out

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """前向传播"""
        import time

        start_time = time.time()

        B, N, C = x.shape

        # 生成Q, K, V
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # 每个的形状: (B, num_heads, N, head_dim)

        # 应用归一化
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 根据后端选择注意力实现
        # if self.backend == 'flash_attn':
        #     attn_output = self._flash_attn_attention(q, k, v, attn_mask)
        # elif self.backend == 'pytorch':
        #     attn_output = self._pytorch_attention(q, k, v, attn_mask)
        # else:  # standard
        #     attn_output = self._standard_attention(q, k, v, attn_mask)

        attn_output = self._flash_attn_attention(q, k, v, attn_mask)

        # 重新整形输出
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)

        # 输出投影
        output = self.proj(attn_output)
        output = self.proj_drop(output)

        # 更新性能统计
        self.performance_stats["forward_calls"] += 1
        self.performance_stats["total_time"] += time.time() - start_time

        return output

    def get_performance_stats(self) -> dict:
        """获取性能统计信息"""
        stats = self.performance_stats.copy()
        if stats["forward_calls"] > 0:
            stats["avg_time_per_call"] = stats["total_time"] / stats["forward_calls"]
        return stats

    def reset_performance_stats(self):
        """重置性能统计"""
        self.performance_stats = {
            "backend_used": self.backend,
            "forward_calls": 0,
            "total_time": 0.0,
        }


class FlashAttnBlock(Block):
    """
    继承自timm Block的Flash Attention Block实现

    这个Block使用Flash Attention替代标准的多头注意力，
    专为视觉任务优化，可以显著提升长序列的训练和推理效率。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        mlp_layer: Type[nn.Module] = Mlp,
        # Flash Attention特有参数
        flash_backend: str = "flash_attn",  # 默认使用flash_attn
        flash_dtype: Optional[torch.dtype] = None,
    ) -> None:
        # 调用父类构造函数，但我们会重写注意力层
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer,
        )

        # 替换默认的注意力层为Flash Attention
        self.attn = FlashAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            backend=flash_backend,
            dtype=flash_dtype,
        )

        # 重新设置LayerScale和DropPath（保持与父类一致）
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # MLP部分保持不变
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, dim)
            attn_mask: 可选的注意力掩码

        Returns:
            输出张量，形状与输入相同
        """
        # 第一个残差连接：注意力
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask)))

        # 第二个残差连接：MLP
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x

    def get_attention_stats(self) -> dict:
        """获取注意力层的性能统计"""
        return self.attn.get_performance_stats()

    def reset_attention_stats(self):
        """重置注意力层的性能统计"""
        self.attn.reset_performance_stats()


# 便利函数和工厂方法
def create_flash_attention_block(
    dim: int = 768,
    num_heads: int = 12,
    mlp_ratio: float = 4.0,
    qkv_bias: bool = True,
    qk_norm: bool = False,
    proj_bias: bool = True,
    proj_drop: float = 0.0,
    attn_drop: float = 0.0,
    drop_path: float = 0.0,
    init_values: Optional[float] = None,
    act_layer: Type[nn.Module] = nn.GELU,
    norm_layer: Type[nn.Module] = nn.LayerNorm,
    flash_backend: str = "flash_attn",  # 默认使用flash_attn
    flash_dtype: Optional[torch.dtype] = None,
) -> FlashAttnBlock:
    """
    创建Flash Attention Block的便利函数

    Args:
        dim: 嵌入维度
        num_heads: 注意力头数
        mlp_ratio: MLP隐藏层维度比例
        qkv_bias: QKV投影是否使用偏置
        qk_norm: 是否对Q和K进行归一化
        proj_bias: 输出投影是否使用偏置
        proj_drop: 投影层dropout概率
        attn_drop: 注意力dropout概率
        drop_path: DropPath概率
        init_values: LayerScale初始值
        act_layer: 激活函数
        norm_layer: 归一化层
        flash_backend: Flash Attention后端选择 (默认: flash_attn)
        flash_dtype: Flash Attention计算精度

    Returns:
        FlashAttnBlock实例
    """
    return FlashAttnBlock(
        dim=dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_norm=qk_norm,
        proj_bias=proj_bias,
        proj_drop=proj_drop,
        attn_drop=attn_drop,
        init_values=init_values,
        drop_path=drop_path,
        act_layer=act_layer,
        norm_layer=norm_layer,
        flash_backend=flash_backend,
        flash_dtype=flash_dtype,
    )


def test_flash_attention_block():
    """测试Flash Attention Block的功能"""
    print("测试Flash Attention Block...")

    # 创建测试数据
    batch_size, seq_len, dim = 2, 512, 768
    num_heads = 12

    x = torch.randn(batch_size, seq_len, dim)

    # 创建Flash Attention Block (默认使用flash_attn后端)
    block = create_flash_attention_block(
        dim=dim, num_heads=num_heads, flash_backend="flash_attn"
    )

    # 前向传播
    with torch.no_grad():
        output = block(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"性能统计: {block.get_attention_stats()}")

    # 测试不同后端
    backends = ["flash_attn", "pytorch", "standard"]
    for backend in backends:
        try:
            test_block = create_flash_attention_block(
                dim=dim, num_heads=num_heads, flash_backend=backend
            )
            with torch.no_grad():
                _ = test_block(x)
            print(f"✅ 后端 '{backend}' 测试成功")
        except Exception as e:
            print(f"❌ 后端 '{backend}' 测试失败: {e}")


if __name__ == "__main__":
    test_flash_attention_block()
