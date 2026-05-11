import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger("flag_gems." + __name__)

# Import the main attention implementation from flag_gems
from flag_gems.ops.attention import (
    ScaleDotProductAttention,
    scaled_dot_product_attention as generic_scaled_dot_product_attention,
)


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    """Metax specialized implementation of scaled dot product attention.

    This is essentially the same Triton-based implementation as the generic version,
    but wrapped with Metax-specific logging.
    """
    logger.debug("METAX GEMS SCALED DOT PRODUCT ATTENTION (GQA=%s)", enable_gqa)

    # Delegate to the generic implementation
    return generic_scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )


def scaled_dot_product_attention_forward(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    """Forward function for scaled dot product attention."""
    logger.debug("METAX GEMS SCALED DOT PRODUCT ATTENTION FORWARD (GQA=%s)", enable_gqa)
    return ScaleDotProductAttention.apply(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        enable_gqa,
    )