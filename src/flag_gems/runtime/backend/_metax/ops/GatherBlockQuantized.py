import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)


@libentry()
@triton.heuristics(
    {
        "BLOCK_SIZE": lambda args: 512,
    }
)
@triton.jit
def gather_kernel(
    input_ptr,
    indices_ptr,
    output_ptr,
    input_row_stride,
    indices_row_stride,
    output_row_stride,
    n_indices,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tle.program_id(axis=0)

    # Calculate the starting row for this program
    row_offset = pid * input_row_stride

    # Load indices for this row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_indices

    # Load the indices
    indices = tl.load(indices_ptr + col_offsets * indices_row_stride, mask=mask, other=0)

    # Gather values from input
    # For each index, load the corresponding value from the input row
    gathered = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(BLOCK_SIZE):
        idx = tl.load(indices_ptr + i * indices_row_stride)
        inp_offset = row_offset + idx * n_cols + tl.arange(0, n_cols)
        val = tl.load(input_ptr + row_offset + idx * n_cols + tl.arange(0, n_cols), mask=tl.arange(0, n_cols) < n_cols, other=0.0)
        gathered = tl.where(i == 0, val, gathered)

    # Store output
    out_row_offset = pid * output_row_stride
    tl.store(output_ptr + out_row_offset + col_offsets * output_row_stride, gathered, mask=mask)


def gather_block_quantized(input_tensor, indices, dimension=0):
    """Gather values from input tensor at specified indices along given dimension.

    This is a fallback implementation using torch gather for block-quantized tensors.
    The actual GatherBlockQuantized operator might have specific behavior for quantized
    tensors that this implementation approximates using the standard gather.

    Args:
        input_tensor: The input tensor to gather from
        indices: The indices to gather
        dimension: The dimension along which to gather

    Returns:
        The gathered tensor
    """
    logger.debug("METAX GEMS GATHER_BLOCK_QUANTIZED")

    # For now, use torch gather as a fallback
    # This handles the case where the actual aten::gather_block_quantized is not available
    return torch.gather(input_tensor, dimension, indices)


# Alias for consistency with naming convention
GatherBlockQuantized = gather_block_quantized