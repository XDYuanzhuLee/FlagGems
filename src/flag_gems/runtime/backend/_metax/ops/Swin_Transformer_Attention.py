import logging
import math

import torch
import torch.nn.functional as F

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger("flag_gems." + __name__)


def swin_transformer_attention(query, key, value, scale=None, attention_mask=None):
    """
    Swin Transformer Attention operator.

    This function computes the attention mechanism used in Swin Transformer.
    For standard (non-windowed) attention, this is equivalent to scaled dot-product
    attention. For true Swin Transformer, window shifting should be applied to
    inputs before calling this function.

    Args:
        query (torch.Tensor): Query tensor of shape [BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM]
        key (torch.Tensor): Key tensor of shape [BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM]
        value (torch.Tensor): Value tensor of shape [BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM]
        scale (float, optional): Scale factor for QK dot product. Defaults to 1/sqrt(HEAD_DIM)
        attention_mask (torch.Tensor, optional): Attention mask tensor

    Returns:
        torch.Tensor: Attention output of shape [BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM]
    """
    logger.debug("METAX GEMS SWIN_TRANSFORMER_ATTENTION")

    # Default scale
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])

    # Transpose for computation: [B, H, S, D] -> [B, S, H, D]
    query_t = query.transpose(1, 2)
    key_t = key.transpose(1, 2)
    value_t = value.transpose(1, 2)

    # Compute attention scores
    qk = torch.matmul(query_t, key_t.transpose(-2, -1)) * scale

    if attention_mask is not None:
        qk = qk + attention_mask

    # Apply softmax
    attn_weights = torch.softmax(qk, dim=-1)

    # Compute output
    output = torch.matmul(attn_weights, value_t)

    # Transpose back: [B, S, H, D] -> [B, H, S, D]
    output = output.transpose(1, 2)

    return output


class _SwinTransformerAttention(torch.autograd.Function):
    """Autograd function for Swin Transformer Attention"""

    @staticmethod
    def forward(ctx, query, key, value, scale=None, attention_mask=None):
        logger.debug("METAX GEMS SWIN_TRANSFORMER_ATTENTION FORWARD")
        output = swin_transformer_attention(query, key, value, scale, attention_mask)
        ctx.save_for_backward(query, key, value)
        ctx.scale = scale
        ctx.attention_mask = attention_mask
        return output

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("METAX GEMS SWIN_TRANSFORMER_ATTENTION BACKWARD")
        query, key, value = ctx.saved_tensors
        scale = ctx.scale if ctx.scale is not None else 1.0 / math.sqrt(query.shape[-1])

        # Transpose for computation
        query_t = query.transpose(1, 2)
        key_t = key.transpose(1, 2)
        value_t = value.transpose(1, 2)
        grad_output_t = grad_output.transpose(1, 2)

        BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM = query_t.shape

        # Compute attention weights
        qk = torch.matmul(query_t, key_t.transpose(-2, -1)) * scale
        if ctx.attention_mask is not None:
            qk = qk + ctx.attention_mask
        attn_weights = torch.softmax(qk, dim=-1)

        # Compute gradients
        grad_attn = torch.matmul(grad_output_t, value_t.transpose(-2, -1))
        grad_v = torch.matmul(attn_weights.transpose(-2, -1), grad_output_t)

        grad_qk = torch.matmul(grad_attn, value_t) * scale
        grad_k = torch.matmul(grad_qk.transpose(-2, -1), query_t)
        grad_q = torch.matmul(grad_qk, key_t)

        # Transpose back
        grad_query = grad_q.transpose(1, 2)
        grad_key = grad_k.transpose(1, 2)
        grad_value = grad_v.transpose(1, 2)

        return grad_query, grad_key, grad_value, None, None


def swin_transformer_attention_autograd(query, key, value, scale=None, attention_mask=None):
    """
    Swin Transformer Attention with autograd support.
    """
    return _SwinTransformerAttention.apply(query, key, value, scale, attention_mask)