from functools import partial

from timm.models.registry import register_model
from timm.models.vision_transformer import (
    VisionTransformer,
    build_model_with_cfg,
    checkpoint_filter_fn,
)

from models.modules.attention.mhla import MHLA_Normed_Torch
from models.modules.timm_block import LinearAttnBlock
from models.modules.timm_block.mhla import MHLA_Uniform_Block
from models.mhla_vit import MHLA_ViT


def _create_deit(
    variant, pretrained=False, distilled=False, model_cls=VisionTransformer, **kwargs
):
    out_indices = kwargs.pop("out_indices", 3)
    model = build_model_with_cfg(
        model_cls,
        variant,
        pretrained,
        pretrained_filter_fn=partial(checkpoint_filter_fn, adapt_layer_scale=True),
        feature_cfg=dict(out_indices=out_indices, feature_cls="getter"),
        **kwargs,
    )
    return model


@register_model
def deit_small_linear(**kwargs):
    del kwargs["pretrained"]
    kwargs["class_token"] = False
    model_args = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        block_fn=LinearAttnBlock,
        norm_layer="rmsnorm",
        qk_norm=True,
        **kwargs,
    )
    model = _create_deit("deit_small_linear", pretrained=False, **model_args)
    return model


@register_model
def deit_base_linear(**kwargs):
    del kwargs["pretrained"]
    kwargs["class_token"] = False
    model_args = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        block_fn=LinearAttnBlock,
        norm_layer="rmsnorm",
        qk_norm=True,
        **kwargs,
    )
    model = _create_deit("deit_base_linear", pretrained=False, **model_args)
    return model


@register_model
def deit_small(**kwargs):
    del kwargs["pretrained"]
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_deit(
        "deit_small_patch16_224", pretrained=False, **dict(model_args, **kwargs)
    )
    return model


@register_model
def deit_small_384(**kwargs):
    del kwargs["pretrained"]
    model_args = dict(patch_size=16, embed_dim=384, depth=12, img_size=384, num_heads=6)
    model = _create_deit(
        "deit_small_384", pretrained=False, **dict(model_args, **kwargs)
    )
    return model


@register_model
def deit_small_linear_384(**kwargs):
    del kwargs["pretrained"]
    kwargs["class_token"] = False
    model_args = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        block_fn=LinearAttnBlock,
        img_size=384,
        norm_layer="rmsnorm",
        qk_norm=True,
        **kwargs,
    )
    model = _create_deit("deit_small_linear_384", pretrained=False, **model_args)
    return model


@register_model
def deit_tiny_pla_1d_v6_6(**kwargs):
    del kwargs["pretrained"]
    assert "piece_size" in kwargs, (
        "Please specify 'piece_size' in kwargs for PLA_1D model."
    )  # default to 49
    assert "transform" in kwargs, (
        "Please specify 'transform' in kwargs for PLA_1D model."
    )
    piece_size = kwargs["piece_size"]
    transform = kwargs["transform"]
    exp_sigma = kwargs["exp_sigma"]
    attn_func = partial(
        MHLA_Normed_Torch,
        window_size=piece_size**2,
        embed_len=256,
        transform=transform,
        exp_sigma=exp_sigma,
    )

    pla_1d_block_kwargs = dict(
        attn_func=attn_func,
    )

    # kwargs['embed_layer'] = partial(Autopad_PatchEmbed, block_size=window_size)
    kwargs["class_token"] = False

    del kwargs["piece_size"]
    del kwargs["transform"]
    del kwargs["exp_sigma"]
    model_args = dict(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        img_size=256,
        qk_norm=True,
        flatten=False,
        piece_size=piece_size,
        block_fn=partial(MHLA_Uniform_Block, **pla_1d_block_kwargs),
        **kwargs,
    )
    return _create_deit(
        "deit_small_pla_1d_v6", pretrained=False, model_cls=MHLA_ViT, **model_args
    )


@register_model
def deit_base_pla_1d_v6_6(**kwargs):
    del kwargs["pretrained"]
    assert "piece_size" in kwargs, (
        "Please specify 'piece_size' in kwargs for PLA_1D model."
    )  # default to 49
    assert "transform" in kwargs, (
        "Please specify 'transform' in kwargs for PLA_1D model."
    )
    piece_size = kwargs["piece_size"]
    transform = kwargs["transform"]
    exp_sigma = kwargs["exp_sigma"]
    attn_func = partial(
        MHLA_Normed_Torch,
        window_size=piece_size**2,
        embed_len=256,
        transform=transform,
        exp_sigma=exp_sigma,
    )

    pla_1d_block_kwargs = dict(
        attn_func=attn_func,
    )

    # kwargs['embed_layer'] = partial(Autopad_PatchEmbed, block_size=window_size)
    kwargs["class_token"] = False

    del kwargs["piece_size"]
    del kwargs["transform"]
    del kwargs["exp_sigma"]
    model_args = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        img_size=256,
        qk_norm=True,
        flatten=False,
        piece_size=piece_size,
        block_fn=partial(MHLA_Uniform_Block, **pla_1d_block_kwargs),
        **kwargs,
    )
    return _create_deit(
        "deit_base_pla_1d_v6", pretrained=False, model_cls=MHLA_ViT, **model_args
    )


@register_model
def deit_small_pla_1d_v6_6(**kwargs):
    del kwargs["pretrained"]
    assert "piece_size" in kwargs, (
        "Please specify 'piece_size' in kwargs for PLA_1D model."
    )  # default to 49
    assert "transform" in kwargs, (
        "Please specify 'transform' in kwargs for PLA_1D model."
    )
    piece_size = kwargs["piece_size"]
    transform = kwargs["transform"]
    exp_sigma = kwargs["exp_sigma"]
    attn_func = partial(
        MHLA_Normed_Torch,
        window_size=piece_size**2,
        embed_len=256,
        transform=transform,
        exp_sigma=exp_sigma,
    )

    pla_1d_block_kwargs = dict(
        attn_func=attn_func,
    )

    # kwargs['embed_layer'] = partial(Autopad_PatchEmbed, block_size=window_size)
    kwargs["class_token"] = False

    del kwargs["piece_size"]
    del kwargs["transform"]
    del kwargs["exp_sigma"]
    model_args = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        img_size=256,
        qk_norm=True,
        flatten=False,
        piece_size=piece_size,
        block_fn=partial(MHLA_Uniform_Block, **pla_1d_block_kwargs),
        **kwargs,
    )
    return _create_deit(
        "deit_small_pla_1d_v6", pretrained=False, model_cls=MHLA_ViT, **model_args
    )
