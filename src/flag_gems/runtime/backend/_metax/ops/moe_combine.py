import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["hidden_size", "topk"],
)
@triton.jit
def moe_combine_kernel(
    input_ptr,
    output_ptr,
    num_tokens,
    topk,
    hidden_size,
    input_stride_token,
    input_stride_topk,
    input_stride_hidden,
    output_stride_token,
    output_stride_hidden,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tle.program_id(0)
    block_idx = tle.program_id(1)
    hidden_start = block_idx * BLOCK_SIZE
    hidden_offsets = hidden_start + tl.arange(0, BLOCK_SIZE)
    hidden_mask = hidden_offsets < hidden_size

    if token_idx >= num_tokens:
        return

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    input_base = input_ptr + token_idx * input_stride_token

    for expert_idx in range(topk):
        expert_ptr = input_base + expert_idx * input_stride_topk
        expert_data = tl.load(expert_ptr + hidden_offsets, mask=hidden_mask, other=0.0)
        acc += expert_data

    output_ptr_pos = output_ptr + token_idx * output_stride_token + hidden_offsets

    tl.store(
        output_ptr_pos,
        acc.to(tl.float16) if input_ptr.dtype.element_ty == tl.float16 else acc,
        mask=hidden_mask,
    )


def moe_combine(input: torch.Tensor, output: torch.Tensor):
    """Combine top-k expert outputs by summing them.

    Args:
        input: Tensor of shape (num_tokens, topk, hidden_size)
        output: Tensor of shape (num_tokens, hidden_size)
    """
    logger.debug("METAX GEMS MOE_COMBINE")
    num_tokens, topk, hidden_size = input.shape
    input_strides = input.stride()
    output_strides = output.stride()

    logger.debug(
        "METAX GEMS MOE_COMBINE, [shape info]: num_tokens=%s, topk=%s, hidden_size=%s",
        num_tokens,
        topk,
        hidden_size,
    )

    grid = lambda meta: (num_tokens, triton.cdiv(hidden_size, meta["BLOCK_SIZE"]))
    with torch_device_fn.device(input.device):
        moe_combine_kernel[grid](
            input,
            output,
            num_tokens,
            topk,
            hidden_size,
            input_strides[0],
            input_strides[1],
            input_strides[2],
            output_strides[0],
            output_strides[1],
        )
    return output