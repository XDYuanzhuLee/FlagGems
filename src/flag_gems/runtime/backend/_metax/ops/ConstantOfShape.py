import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger("flag_gems." + __name__)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": k}, num_warps=w, num_stages=4)
        for w in [2, 4, 8, 16]
        for k in [1024, 2048, 4096, 8192]
    ],
    key=[
        "N",
    ],
)
@triton.jit()
def constant_of_shape_kernel(
    output_ptr,
    N,
    fill_value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    tl.store(output_ptr + offsets, fill_value, mask=mask)


def constant_of_shape(shape_tensor, fill_value=0, *, dtype=None, device=None):
    """
    Creates a tensor filled with a constant value.

    Args:
        shape_tensor: A 1D tensor containing the shape of the output tensor.
        fill_value: The value to fill the output tensor with. Default is 0.
        dtype: The data type of the output tensor. If None, defaults to float32.
        device: The device on which to create the output tensor.
    """
    logger.debug("METAX GEMS CONSTANT OF SHAPE")

    # Extract shape from tensor
    shape = shape_tensor.tolist()

    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.get_default_dtype()

    # Handle scalar tensor case
    if len(shape) == 0:
        shape = []

    out = torch.empty(shape, device=device, dtype=dtype)
    N = volume(shape)

    # If the tensor is empty, just return
    if N == 0:
        return out

    # Convert fill_value to the appropriate dtype
    if isinstance(fill_value, bool):
        if dtype != torch.bool:
            fill_value = int(fill_value)
    elif isinstance(fill_value, int):
        if dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            fill_value = float(fill_value)
    elif isinstance(fill_value, float):
        if dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            fill_value = int(fill_value)

    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(device):
        constant_of_shape_kernel[grid_fn](
            out,
            N,
            fill_value,
        )
    return out