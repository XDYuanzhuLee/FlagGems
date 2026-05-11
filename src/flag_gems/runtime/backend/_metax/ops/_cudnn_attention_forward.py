import logging

import torch

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger("flag_gems." + __name__)


def _cudnn_attention_forward(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    """CuDNN attention forward implementation for Metax backend.

    This is a wrapper around PyTorch's scaled_dot_product_attention with
    cuDNN backend enabled.

    Args:
        query: Query tensor of shape (batch, num_heads, seq_len, head_dim)
        key: Key tensor of shape (batch, num_kv_heads, seq_len, head_dim)
        value: Value tensor of shape (batch, num_kv_heads, seq_len, head_dim)
        attn_mask: Optional attention mask
        dropout_p: Dropout probability (currently must be 0.0)
        is_causal: Whether to use causal masking
        scale: Optional scale factor
        enable_gqa: Whether to enable grouped-query attention

    Returns:
        Output tensor of shape (batch, num_heads, seq_len, head_dim)
    """
    logger.debug("METAX GEMS CUDNN ATTENTION FORWARD")

    # Ensure dropout is 0.0 for now
    assert dropout_p == 0.0, "Currently only support dropout_p=0.0"

    # Compute scale if not provided
    if scale is None:
        scale = 1.0 / (query.shape[-1] ** 0.5)

    # Use torch's scaled_dot_product_attention which will use cuDNN when available
    # For Metax backend, we delegate to PyTorch's implementation
    with torch_device_fn.device(query.device):
        output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    return output