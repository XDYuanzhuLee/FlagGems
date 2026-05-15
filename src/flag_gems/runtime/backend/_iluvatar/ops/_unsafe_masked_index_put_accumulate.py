import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _unsafe_masked_index_put_accumulate_kernel_1d(
    input_ptr,
    mask_ptr,
    indices_ptr,
    values_ptr,
    input_stride0,
    mask_stride0,
    indices_stride0,
    values_stride0,
    input_size0,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_offset = pid * BLOCK_SIZE
    offsets = block_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load mask
    mask_value = tl.load(mask_ptr + offsets * mask_stride0, mask=mask, other=0)

    # Only process where mask is True
    active_mask = mask_value != 0

    # Get index and value for active positions
    idx = tl.load(indices_ptr + offsets * indices_stride0, mask=mask, other=0)
    val = tl.load(values_ptr + offsets * values_stride0, mask=mask, other=0)

    # Compute input offset
    input_offset = idx * input_stride0

    # Atomic add for accumulation (only where mask is True)
    tl.atomic_add(input_ptr + input_offset, val, mask=active_mask)


@libentry()
@triton.jit
def _unsafe_masked_index_put_accumulate_kernel_2d(
    input_ptr,
    mask_ptr,
    indices0_ptr,
    indices1_ptr,
    values_ptr,
    input_stride0,
    input_stride1,
    mask_stride0,
    mask_stride1,
    indices_stride0,
    indices_stride1,
    values_stride0,
    values_stride1,
    input_size0,
    input_size1,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_offset = pid * BLOCK_SIZE
    offsets = block_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Convert flat offset to 2D coordinates
    row = offsets // input_size1
    col = offsets % input_size1

    # Compute flat offsets for each array
    mask_offset = row * mask_stride0 + col * mask_stride1
    indices_offset = row * indices_stride0 + col * indices_stride1
    values_offset = row * values_stride0 + col * values_stride1

    # Load mask
    mask_value = tl.load(mask_ptr + mask_offset, mask=mask, other=0)

    # Only process where mask is True
    active_mask = mask_value != 0

    # Load indices using 2D coordinates
    idx0 = tl.load(indices0_ptr + indices_offset, mask=mask, other=0)
    idx1 = tl.load(indices1_ptr + indices_offset, mask=mask, other=0)

    # Load values using 2D coordinates
    val = tl.load(values_ptr + values_offset, mask=mask, other=0)

    # Compute input offset
    input_offset = idx0 * input_stride0 + idx1 * input_stride1

    # Atomic add for accumulation
    tl.atomic_add(input_ptr + input_offset, val, mask=active_mask)


@libentry()
@triton.jit
def _unsafe_masked_index_put_accumulate_kernel_3d(
    input_ptr,
    mask_ptr,
    indices0_ptr,
    indices1_ptr,
    indices2_ptr,
    values_ptr,
    input_stride0,
    input_stride1,
    input_stride2,
    mask_stride0,
    mask_stride1,
    mask_stride2,
    indices_stride0,
    indices_stride1,
    indices_stride2,
    values_stride0,
    values_stride1,
    values_stride2,
    input_size0,
    input_size1,
    input_size2,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_offset = pid * BLOCK_SIZE
    offsets = block_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Convert flat offset to 3D coordinates
    plane_size = input_size1 * input_size2
    plane = offsets // plane_size
    remainder = offsets % plane_size
    row = remainder // input_size2
    col = remainder % input_size2

    # Compute flat offsets for each array
    mask_offset = plane * mask_stride0 + row * mask_stride1 + col * mask_stride2
    indices_offset = plane * indices_stride0 + row * indices_stride1 + col * indices_stride2
    values_offset = plane * values_stride0 + row * values_stride1 + col * values_stride2

    # Load mask
    mask_value = tl.load(mask_ptr + mask_offset, mask=mask, other=0)

    # Only process where mask is True
    active_mask = mask_value != 0

    # Load indices
    idx0 = tl.load(indices0_ptr + indices_offset, mask=mask, other=0)
    idx1 = tl.load(indices1_ptr + indices_offset, mask=mask, other=0)
    idx2 = tl.load(indices2_ptr + indices_offset, mask=mask, other=0)

    # Load values
    val = tl.load(values_ptr + values_offset, mask=mask, other=0)

    # Compute input offset
    input_offset = idx0 * input_stride0 + idx1 * input_stride1 + idx2 * input_stride2

    # Atomic add for accumulation
    tl.atomic_add(input_ptr + input_offset, val, mask=active_mask)


def _unsafe_masked_index_put_accumulate(input, mask, indices, values):
    logger.debug("ILUVATAR GEMS _UNSAFE_MASKED_INDEX_PUT_ACCUMULATE")

    input = input.clone()
    device = input.device
    input_shape = input.shape
    ndim = input.ndim

    # Get total number of elements to process
    N = mask.numel()

    # Define grid
    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    BLOCK_SIZE = 128

    if ndim == 1:
        _unsafe_masked_index_put_accumulate_kernel_1d[grid](
            input,
            mask,
            indices[0],
            values,
            input.stride(0),
            mask.stride(0),
            indices[0].stride(0),
            values.stride(0),
            input_shape[0],
            N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    elif ndim == 2:
        _unsafe_masked_index_put_accumulate_kernel_2d[grid](
            input,
            mask,
            indices[0],
            indices[1],
            values,
            input.stride(0),
            input.stride(1),
            mask.stride(0),
            mask.stride(1),
            indices[0].stride(0),
            indices[0].stride(1),
            values.stride(0),
            values.stride(1),
            input_shape[0],
            input_shape[1],
            N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    elif ndim == 3:
        _unsafe_masked_index_put_accumulate_kernel_3d[grid](
            input,
            mask,
            indices[0],
            indices[1],
            indices[2],
            values,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            mask.stride(0),
            mask.stride(1),
            mask.stride(2),
            indices[0].stride(0),
            indices[0].stride(1),
            indices[0].stride(2),
            values.stride(0),
            values.stride(1),
            values.stride(2),
            input_shape[0],
            input_shape[1],
            input_shape[2],
            N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # For higher dimensions, use torch's implementation
        return torch._unsafe_masked_index_put_accumulate(input, mask, indices, values)

    return input