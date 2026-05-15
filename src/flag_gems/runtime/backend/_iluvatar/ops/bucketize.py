import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry, pointwise_dynamic

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def bucketize_kernel(
    input_ptr,
    boundaries_ptr,
    output_ptr,
    input_numel,
    boundaries_numel,
    right: tl.constexpr,
    is_int32: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = block_start + offsets < input_numel
    input_block_ptrs = input_ptr + (block_start + offsets)
    input_vals = tl.load(input_block_ptrs, mask=mask, other=0.0)

    boundaries_start_ptr = boundaries_ptr
    boundaries_offsets = tl.arange(0, BLOCK_SIZE)
    boundaries_block_ptrs = boundaries_start_ptr + boundaries_offsets * boundaries_numel.dtype.bit_length

    if is_int32:
        output_type = tl.int32
    else:
        output_type = tl.int64

    if right:
        result = bucketize_right(input_vals, boundaries_ptr, boundaries_numel, output_type)
    else:
        result = bucketize_left(input_vals, boundaries_ptr, boundaries_numel, output_type)

    output_block_ptrs = output_ptr + (block_start + offsets)
    tl.store(output_block_ptrs, result, mask=mask)


@triton.jit
def bucketize_left(val, boundaries_ptr, numel, output_type):
    n = numel
    if n == 0:
        return tl.zeros(1, output_type)

    if val <= tl.load(boundaries_ptr):
        return tl.zeros(1, output_type)
    if val > tl.load(boundaries_ptr + (n - 1)):
        return tl.full((1,), n, output_type)

    lo = 0
    hi = n

    for _ in range(20):
        mid = (lo + hi) // 2
        mid_val = tl.load(boundaries_ptr + mid)
        if val > mid_val:
            lo = mid + 1
        else:
            hi = mid

    result = lo
    return tl.full((1,), result, output_type)


@triton.jit
def bucketize_right(val, boundaries_ptr, numel, output_type):
    n = numel
    if n == 0:
        return tl.zeros(1, output_type)

    if val < tl.load(boundaries_ptr):
        return tl.zeros(1, output_type)
    if val >= tl.load(boundaries_ptr + (n - 1)):
        return tl.full((1,), n, output_type)

    lo = 0
    hi = n

    for _ in range(20):
        mid = (lo + hi) // 2
        mid_val = tl.load(boundaries_ptr + mid)
        if val >= mid_val:
            lo = mid + 1
        else:
            hi = mid

    result = lo
    return tl.full((1,), result, output_type)


def bucketize(input, boundaries, *, right=False, out_int32=False):
    logger.debug("ILUVATAR GEMS BUCKETIZE")

    if not isinstance(input, torch.Tensor):
        input = torch.tensor(input, device=boundaries.device)
    if not isinstance(boundaries, torch.Tensor):
        boundaries = torch.tensor(boundaries, device=input.device)

    if boundaries.numel() == 0:
        return torch.zeros_like(input, dtype=torch.int32 if out_int32 else torch.int64)

    output_dtype = torch.int32 if out_int32 else torch.int64

    if input.numel() == 0:
        return torch.empty_like(input, dtype=output_dtype)

    input = input.contiguous()
    boundaries = boundaries.contiguous()

    output = torch.empty_like(input, dtype=output_dtype)

    BLOCK_SIZE = 128
    grid = ((input.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    bucketize_kernel[grid](
        input,
        boundaries,
        output,
        input.numel(),
        boundaries.numel(),
        right,
        out_int32,
        BLOCK_SIZE,
    )

    return output