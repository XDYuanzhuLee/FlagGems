import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def renorm_kernel(
    input_ptr,
    output_ptr,
    p,
    dim,
    maxnorm,
    ndim: tl.constexpr,
    input_strides: tl.constexpr,
    output_strides: tl.constexpr,
    input_shape: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Renorm normalizes sub-tensors along dimension `dim` such that their p-norm <= maxnorm
    # For each sub-tensor (fixed indices except along dim), compute the p-norm
    # If norm > maxnorm, scale all elements by maxnorm / norm

    pid = tl.program_id(0)
    num_sub_tensors = input_shape[0]  # total number of sub-tensors (product of all dims except dim)
    if pid >= num_sub_tensors:
        return

    # Reconstruct the index into the sub-tensor
    # We need to compute the multi-dimensional index from pid
    idx = pid
    indices = []
    remaining = idx
    for i in range(ndim):
        stride = 1
        for j in range(i + 1, ndim):
            stride *= input_shape[j]
        indices.append(remaining // stride)
        remaining = remaining % stride

    # Now indices contains the multi-dimensional index
    # We need to compute the starting offset for this sub-tensor
    offset = 0
    for i in range(ndim):
        if i != dim:
            offset += indices[i] * input_strides[i]

    # Now iterate along the dim dimension and compute p-norm
    p_scalar = p.to(tl.float32)
    maxnorm_scalar = maxnorm.to(tl.float32)

    norm = 0.0
    # Load and compute norm
    for i in range(0, input_shape[dim], BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < input_shape[dim]
        full_offset = offset + offs * input_strides[dim]
        vals = tl.load(input_ptr + full_offset, mask=mask, other=0.0)
        # Compute |x|^p
        abs_vals = tl.abs(vals)
        powered = tl_extra_shim.pow(abs_vals, p_scalar)
        norm += tl.sum(powered, mask=mask)

    # Compute final norm: norm^(1/p)
    norm = tl_extra_shim.pow(norm, 1.0 / p_scalar)

    # Compute scaling factor
    scale = tl.where(norm > maxnorm_scalar, maxnorm_scalar / norm, 1.0)

    # Now apply scaling and store result
    for i in range(0, input_shape[dim], BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < input_shape[dim]
        full_offset = offset + offs * input_strides[dim]
        vals = tl.load(input_ptr + full_offset, mask=mask, other=0.0)
        scaled_vals = vals * scale
        tl.store(output_ptr + full_offset, scaled_vals, mask=mask)


def renorm(input: torch.Tensor, p: float, dim: int, maxnorm: float) -> torch.Tensor:
    logger.debug("ILUVATAR GEMS RENORM")

    output = torch.empty_like(input)

    # Handle negative dim
    if dim < 0:
        dim = input.dim() + dim

    ndim = input.dim()
    # Flatten the tensor to get the number of sub-tensors
    # We need to iterate over all dimensions except dim

    # Compute total number of sub-tensors
    num_sub_tensors = 1
    for i in range(ndim):
        if i != dim:
            num_sub_tensors *= input.shape[i]

    # Get strides
    input_strides = input.stride()
    output_strides = output.stride()
    input_shape = input.shape

    # Define block size
    BLOCK_SIZE = 1024

    # Launch kernel
    grid = (num_sub_tensors,)

    with torch.cuda.device(input.device):
        renorm_kernel[grid](
            input,
            output,
            p,
            dim,
            maxnorm,
            ndim,
            input_strides,
            output_strides,
            input_shape,
            BLOCK_SIZE,
        )

    return output