import logging

import torch

from flag_gems.ops.attention import scaled_dot_product_attention_backward

logger = logging.getLogger("flag_gems." + __name__)


def _efficient_attention_backward(
    grad_out,
    query,
    key,
    value,
    bias,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    logsumexp,
    dropout_p,
    philox_seed,
    philox_offset,
    custom_mask_type,
    bias_requires_grad,
    *,
    scale=None,
    num_splits_key=None,
    window_size=None,
    shared_storage_dqdkdv=False,
):
    """
    Metax specialized implementation of _efficient_attention_backward.
    This uses the existing scaled_dot_product_attention_backward from attention.py
    which is implemented using Triton kernels.
    """
    logger.debug("METAX GEMS _EFFICIENT_ATTENTION_BACKWARD")

    # Extract scale from query if not provided
    if scale is None:
        scale = 1.0 / (query.shape[-1] ** 0.5)

    # Call the existing scaled_dot_product_attention_backward
    # This uses Triton kernels from attention.py
    dq, dk, dv = scaled_dot_product_attention_backward(
        do=grad_out,
        query=query,
        key=key,
        value=value,
        o=out,
        M=logsumexp,
        attn_mask=bias,
        dropout_p=dropout_p,
        is_causal=False,
        scale=scale,
        enable_gqa=False,
    )

    # Return d_query, d_key, d_value, d_bias (bias gradient is None since we don't track it)
    return dq, dk, dv, None