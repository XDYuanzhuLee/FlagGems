import logging

import torch

from flag_gems import scaled_dot_product_attention as gems_sdpa

logger = logging.getLogger("flag_gems." + __name__)


def _scaled_dot_product_attention_math_for_mps(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = None,
    enable_gqa: bool = False,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """Metax specialized implementation of _scaled_dot_product_attention_math_for_mps.

    This function provides a Metax-specific implementation for the PyTorch MPS attention
    operator, which is not available for CUDA backend. It delegates to FlagGems'
    scaled_dot_product_attention implementation.

    Args:
        query: Query tensor of shape (batch, num_heads, seq_len_q, head_dim)
        key: Key tensor of shape (batch, num_kv_heads, seq_len_kv, head_dim)
        value: Value tensor of shape (batch, num_kv_heads, seq_len_kv, head_dim)
        attn_mask: Optional attention mask tensor
        dropout_p: Dropout probability (currently only 0.0 is supported)
        is_causal: Whether to apply causal masking
        scale: Optional scale factor (defaults to 1/sqrt(head_dim))
        enable_gqa: Enable grouped-query attention
        bias: Optional bias tensor (alias for attn_mask for compatibility)

    Returns:
        Output tensor of shape (batch, num_heads, seq_len_q, head_dim)
    """
    logger.debug("METAX GEMS SCALED_DOT_PRODUCT_ATTENTION_MATH_FOR_MPS")

    # Handle bias parameter as alias for attn_mask
    if bias is not None and attn_mask is None:
        attn_mask = bias

    # Delegate to FlagGems' scaled_dot_product_attention
    output = gems_sdpa(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )

    return output