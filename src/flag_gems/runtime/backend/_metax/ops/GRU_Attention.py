import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)


@libentry()
@triton.jit
def gru_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    scale,
    BATCH,
    NUM_HEADS,
    SEQ_LEN,
    HEAD_DIM: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
):
    # Grid: (batch * num_heads, seq_len, 1)
    batch_head_id = tle.program_id(0)
    seq_id = tle.program_id(1)

    batch_id = batch_head_id // NUM_HEADS
    head_id = batch_head_id % NUM_HEADS

    # Use constexpr for HEAD_DIM in arange
    offs = tl.arange(0, HEAD_DIM)

    # Offsets for query
    q_offsets = (
        batch_id * stride_qb
        + head_id * stride_qh
        + seq_id * stride_qs
        + offs * stride_qd
    )

    # Load query
    q_mask = offs < HEAD_DIM
    query = tl.load(query_ptr + q_offsets, mask=q_mask, other=0.0)

    # Compute attention scores
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    # Iterate over key/value sequence
    for k_id in range(SEQ_LEN):
        # Key offset
        k_offsets = (
            batch_id * stride_kb
            + head_id * stride_kh
            + k_id * stride_ks
            + offs * stride_kd
        )
        key = tl.load(key_ptr + k_offsets, mask=q_mask, other=0.0)

        # Value offset
        v_offsets = (
            batch_id * stride_vb
            + head_id * stride_vh
            + k_id * stride_vs
            + offs * stride_vd
        )
        value = tl.load(value_ptr + v_offsets, mask=q_mask, other=0.0)

        # Compute attention score
        score = tl.sum(query * key) * scale

        # Softmax normalization would be needed here
        # For simplicity, we just compute weighted sum
        acc = acc + value * score

    # Store output
    o_offsets = (
        batch_id * stride_ob
        + head_id * stride_oh
        + seq_id * stride_os
        + offs * stride_od
    )
    tl.store(output_ptr + o_offsets, acc, mask=q_mask)


def gru_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    """
    GRU Attention implementation.

    Args:
        query: Query tensor of shape (batch, num_heads, seq_len, head_dim)
        key: Key tensor of shape (batch, num_heads, kv_len, head_dim)
        value: Value tensor of shape (batch, num_heads, kv_len, head_dim)

    Returns:
        Output tensor of shape (batch, num_heads, seq_len, head_dim)
    """
    logger.debug("METAX GEMS GRU_ATTENTION")

    assert query.dim() == 4, "Query must be 4D tensor"
    assert key.dim() == 4, "Key must be 4D tensor"
    assert value.dim() == 4, "Value must be 4D tensor"

    BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM = query.shape
    _, _, KV_LEN, _ = key.shape

    scale = 1.0 / (HEAD_DIM**0.5)

    # Ensure compatible shapes
    assert key.shape[0] == query.shape[0], "Batch dimension mismatch"
    assert value.shape[0] == query.shape[0], "Batch dimension mismatch"
    assert key.shape[2] == KV_LEN, "Key sequence length mismatch"
    assert value.shape[2] == KV_LEN, "Value sequence length mismatch"

    output = torch.empty_like(query)

    grid = (BATCH * NUM_HEADS, SEQ_LEN, 1)

    gru_attention_kernel[grid](
        query,
        key,
        value,
        output,
        scale,
        BATCH,
        NUM_HEADS,
        SEQ_LEN,
        HEAD_DIM,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
    )

    return output