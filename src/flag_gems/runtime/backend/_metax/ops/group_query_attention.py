import logging

import torch

from flag_gems.ops.attention import scaled_dot_product_attention
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger("flag_gems." + __name__)


def group_query_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = None,
    enable_gqa: bool = True,
):
    """GroupQueryAttention operator for Metax backend.

    This is a wrapper around scaled_dot_product_attention with GQA enabled by default.
    GroupQueryAttention (GQA) is an attention mechanism where multiple query heads
    share a smaller number of key/value heads, reducing memory and compute requirements.

    Args:
        query: Query tensor of shape (batch, num_query_heads, seq_len, head_dim)
        key: Key tensor of shape (batch, num_kv_heads, kv_seq_len, head_dim)
        value: Value tensor of shape (batch, num_kv_heads, kv_seq_len, head_dim)
        attn_mask: Optional attention mask
        dropout_p: Dropout probability (currently must be 0.0)
        is_causal: Whether to apply causal masking
        scale: Optional scale factor for attention scores
        enable_gqa: Whether to enable grouped query attention (default: True)

    Returns:
        Output tensor of shape (batch, num_query_heads, seq_len, head_dim)
    """
    logger.debug("METAX GEMS GROUP_QUERY_ATTENTION")
    # GQA is always enabled for GroupQueryAttention
    with torch_device_fn.device(query.device):
        output = scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=True,  # Always enable GQA for GroupQueryAttention
        )
    return output