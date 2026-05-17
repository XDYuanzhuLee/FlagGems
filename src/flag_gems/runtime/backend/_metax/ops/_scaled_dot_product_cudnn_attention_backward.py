import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger("flag_gems." + __name__)


def scaled_dot_product_cudnn_attention_backward(
    grad_out,
    query,
    key,
    value,
    out,
    logsumexp,
    philox_seed,
    philox_offset,
    attn_bias,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    logger.debug("METAX GEMS SCALED_DOT_PRODUCT_CUDNN_ATTENTION_BACKWARD")

    # Try to use torch's cuDNN implementation first
    try:
        result = torch.ops.aten._scaled_dot_product_cudnn_attention_backward(
            grad_out,
            query,
            key,
            value,
            out,
            logsumexp,
            philox_seed,
            philox_offset,
            attn_bias,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            dropout_p,
            is_causal,
            scale=scale,
        )
        logger.debug("METAX GEMS Using cuDNN implementation")
        return result
    except Exception as e:
        logger.debug(f"METAX GEMS cuDNN not available, using fallback: {e}")

        # Fallback: Use autograd's backward computation
        # This computes the gradients using PyTorch's autograd mechanism
        # We need to recreate the computation graph for the backward pass

        # Ensure inputs require grad for backward
        query = query.requires_grad_(True) if not query.requires_grad else query
        key = key.requires_grad_(True) if not key.requires_grad else key
        value = value.requires_grad_(True) if not value.requires_grad else value

        # Compute scale if not provided
        if scale is None:
            scale = 1.0 / (query.shape[-1] ** 0.5)

        # Use torch's scaled_dot_product_attention which supports autograd
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_bias,
            scale=scale,
            is_causal=is_causal,
            dropout_p=dropout_p,
        )

        # Backward pass using torch autograd
        # Create a dummy loss for backward
        # We need to match grad_out shape to output shape
        if grad_out.shape != output.shape:
            # Handle potential shape mismatches
            grad_out = grad_out.contiguous()

        # Use torch's autograd to compute gradients
        grad_tuple = torch.autograd.grad(
            outputs=output,
            inputs=(query, key, value),
            grad_outputs=(grad_out,),
            retain_graph=True,
            allow_unused=True,
        )

        dq, dk, dv = grad_tuple

        # Handle None gradients
        dq = dq if dq is not None else torch.zeros_like(query)
        dk = dk if dk is not None else torch.zeros_like(key)
        dv = dv if dv is not None else torch.zeros_like(value)

        logger.debug("METAX GEMS Using autograd fallback")
        return (dq, dk, dv)