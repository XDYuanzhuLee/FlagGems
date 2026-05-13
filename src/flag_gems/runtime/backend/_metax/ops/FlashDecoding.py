import logging

import torch

from flag_gems import utils

logger = logging.getLogger("flag_gems." + __name__)


def flash_decoding(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    """
    FlashDecoding operator for Metax backend.

    This is a wrapper around scaled_dot_product_attention that provides
    FlashAttention-like functionality when native flash attention is not available.

    Args:
        query: Query tensor of shape (batch, num_heads, seq_len_q, head_dim)
        key: Key tensor of shape (batch, num_heads_k, seq_len_k, head_dim)
        value: Value tensor of shape (batch, num_heads_k, seq_len_k, head_dim)
        attn_mask: Optional attention mask
        dropout_p: Dropout probability (0.0 means no dropout)
        is_causal: Whether to apply causal masking
        scale: Optional scale factor for attention scores

    Returns:
        Output tensor of shape (batch, num_heads, seq_len_q, head_dim)
    """
    logger.debug("METAX GEMS FLASHDECODING")

    # Validate input shapes and dtypes
    q_dtype = query.dtype
    assert q_dtype in (torch.float16, torch.bfloat16), (
        "FlashDecoding only supports fp16 and bf16 data types"
    )
    assert q_dtype == key.dtype == value.dtype, "Query, key, value must have same dtype"

    # Ensure contiguous memory layout
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # Use scaled_dot_product_attention as the backend
    # This provides FlashAttention-like functionality
    output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p if dropout_p > 0 else 0.0,
        is_causal=is_causal,
        scale=scale,
    )

    return output


def flash_decoding_forward(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    """
    Forward function for FlashDecoding.
    """
    return flash_decoding(query, key, value, attn_mask, dropout_p, is_causal, scale)