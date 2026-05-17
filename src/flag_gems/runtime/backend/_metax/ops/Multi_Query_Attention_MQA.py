import logging

import torch

from flag_gems.ops.attention import (
    ScaleDotProductAttention,
    flash_attention_forward,
    flash_attn_varlen_func,
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
    scaled_dot_product_attention_forward,
)

logger = logging.getLogger("flag_gems." + __name__)


def scaled_dot_product_attention_mqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor = None,
    dropout_p: float = 0.0,
    is_causal: bool = True,
    scale: float = None,
    enable_gqa: bool = True,
) -> torch.Tensor:
    """Multi-Query Attention specialized implementation for Metax backend.

    This function wraps the generic scaled_dot_product_attention with Metax-specific logging.
    Supports Multi-Query Attention (MQA) where key and value have a single head.
    """
    logger.debug("METAX GEMS SCALED_DOT_PRODUCT_ATTENTION_MQA")
    return scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )


def flash_attention_forward_mqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor = None,
    dropout_p: float = 0.0,
    is_causal: bool = True,
    scale: float = None,
) -> torch.Tensor:
    """Flash Attention forward specialized implementation for Metax backend.

    This function wraps the generic flash_attention_forward with Metax-specific logging.
    """
    logger.debug("METAX GEMS FLASH_ATTENTION_FORWARD_MQA")
    return flash_attention_forward(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


__all__ = [
    "scaled_dot_product_attention",
    "scaled_dot_product_attention_backward",
    "scaled_dot_product_attention_forward",
    "scaled_dot_product_attention_mqa",
    "flash_attention_forward",
    "flash_attention_forward_mqa",
    "flash_attn_varlen_func",
    "ScaleDotProductAttention",
]