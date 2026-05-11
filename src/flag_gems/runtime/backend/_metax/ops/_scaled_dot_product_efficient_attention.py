import logging

import torch

from flag_gems.ops.attention import scaled_dot_product_attention_forward

logger = logging.getLogger("flag_gems." + __name__)


def _scaled_dot_product_efficient_attention(
    query,
    key,
    value,
    attn_bias=None,
    compute_log_sumexp=False,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    """
    Implements scaled dot product attention with efficient memory usage.
    This is a metax specialized version based on the standard scaled_dot_product_attention.

    Args:
        query: (batch, num_heads, seq_len_q, head_dim)
        key: (batch, num_kv_heads, seq_len_kv, head_dim)
        value: (batch, num_kv_heads, seq_len_kv, head_dim)
        attn_bias: optional attention bias
        compute_log_sumexp: whether to return log_sumexp for backward
        dropout_p: dropout probability (currently must be 0.0)
        is_causal: whether to use causal masking
        scale: optional scale factor

    Returns:
        output: (batch, num_heads, seq_len_q, head_dim)
        log_sumexp: (batch, num_heads, seq_len_q) if compute_log_sumexp is True
        philox_seed: random seed for dropout
        philox_offset: random offset for dropout
    """
    logger.debug("METAX GEMS SCALED_DOT_PRODUCT_EFFICIENT_ATTENTION")

    # Currently only support dropout_p=0.0
    assert dropout_p == 0.0, "Currently only support dropout_p=0.0"

    # Call the standard attention forward
    output, M = scaled_dot_product_attention_forward(
        query,
        key,
        value,
        attn_bias,
        dropout_p,
        is_causal,
        scale,
        enable_gqa=False,
    )

    # Get the scale if not provided
    if scale is None:
        sm_scale = 1.0 / (query.shape[-1] ** 0.5)
    else:
        sm_scale = scale

    # Compute log_sumexp
    if compute_log_sumexp:
        # M contains the max values, we need to compute log_sumexp
        # log_sumexp = log(sum(exp(qk - max))) + max = log(sum(exp(qk * scale))) / scale
        log_sumexp = M  # M already contains log(sum(exp)) from the attention kernel

        # philox_seed and philox_offset are used for dropout
        # Since dropout_p=0.0, we can return zeros
        philox_seed = torch.tensor(0, dtype=torch.long, device=query.device)
        philox_offset = torch.tensor(0, dtype=torch.long, device=query.device)

        return output, log_sumexp, philox_seed, philox_offset
    else:
        # If compute_log_sumexp is False, still return dummy values for compatibility
        # Some code may expect all 4 return values
        dummy_log_sumexp = torch.empty(
            (query.shape[0], query.shape[1], query.shape[2]),
            device=query.device,
            dtype=torch.float32,
        )
        philox_seed = torch.tensor(0, dtype=torch.long, device=query.device)
        philox_offset = torch.tensor(0, dtype=torch.long, device=query.device)

        return output, dummy_log_sumexp, philox_seed, philox_offset