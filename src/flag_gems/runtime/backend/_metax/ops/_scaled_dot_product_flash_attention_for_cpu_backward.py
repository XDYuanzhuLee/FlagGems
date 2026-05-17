import logging

import torch

logger = logging.getLogger("flag_gems." + __name__)


def scaled_dot_product_flash_attention_for_cpu_backward(
    grad_out,
    query,
    key,
    value,
    out,
    logsumexp,
    dropout_p=0.0,
    is_causal=False,
    attn_mask=None,
    scale=None,
):
    logger.debug(
        "METAX GEMS SCALED_DOT_PRODUCT_FLASH_ATTENTION_FOR_CPU_BACKWARD"
    )

    grad_query, grad_key, grad_value = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu_backward(
        grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        dropout_p,
        is_causal,
        attn_mask=attn_mask,
        scale=scale,
    )
    return grad_query, grad_key, grad_value