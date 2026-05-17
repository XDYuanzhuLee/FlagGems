import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger("flag_gems." + __name__)


@libentry()
@triton.heuristics({"BLOCK_SIZE": lambda _: 1024})
@triton.jit
def _resize_output_kernel(
    input_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, val, mask=mask)


def _resize_output(inp, size, device):
    logger.debug("METAX GEMS _RESIZE_OUTPUT")
    # Create output tensor with target size and initialize with zeros
    out = torch.zeros(size, device=device, dtype=inp.dtype)
    # Calculate number of elements to copy
    numel = min(inp.numel(), out.numel())
    if numel > 0:
        grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
        with torch_device_fn.device(inp.device):
            _resize_output_kernel[grid](inp, out, numel)
    return out


def _resize_output_(inp, size, device):
    logger.debug("METAX GEMS _RESIZE_OUTPUT_")
    # Inplace version: resize the tensor in place
    # Create a new tensor with the target size and copy data
    out = torch.zeros(size, device=device, dtype=inp.dtype)
    numel = min(inp.numel(), out.numel())
    if numel > 0:
        grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
        with torch_device_fn.device(inp.device):
            _resize_output_kernel[grid](inp, out, numel)
    # Copy data back to original tensor
    inp.resize_(size)
    inp.copy_(out)
    return inp