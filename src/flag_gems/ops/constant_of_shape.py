import logging
import math

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.utils.pointwise_dynamic import pointwise_dynamic
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger("flag_gems." + __name__)


ALL_INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64)
ALL_FLOAT_DTYPES = (torch.bfloat16, torch.float16, torch.float32, torch.float64)


def check_dtype(fill_value, dtype, device):
    if isinstance(fill_value, bool):
        if dtype != torch.bool:
            fill_value = int(fill_value)
    elif (
        dtype in ALL_INT_DTYPES
        and (fill_value < torch.iinfo(dtype).min or fill_value > torch.iinfo(dtype).max)
    ) or (
        dtype in ALL_FLOAT_DTYPES
        and not (math.isinf(fill_value) or math.isnan(fill_value))
        and (fill_value < torch.finfo(dtype).min or fill_value > torch.finfo(dtype).max)
    ):
        raise RuntimeError(
            f"value cannot be converted to type {dtype} without overflow"
        )
    if dtype == torch.float64:
        fill_value = torch.tensor(fill_value, dtype=dtype, device=device)
    return fill_value


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def constant_of_shape_func(out, fill_value):
    return fill_value


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def constant_of_shape_func_scalar(out, fill_value):
    return tl.full(out.shape, fill_value, out.dtype)


def constant_of_shape(shape_tensor, fill_value=0, *, dtype=None, device=None):
    """
    Creates a tensor filled with a constant value.

    Args:
        shape_tensor: A 1D tensor containing the shape of the output tensor.
        fill_value: The value to fill the output tensor with. Default is 0.
        dtype: The data type of the output tensor. If None, defaults to float32.
        device: The device on which to create the output tensor.
    """
    logger.debug("GEMS CONSTANT OF SHAPE")

    if device is None:
        device = flag_gems.device
    if dtype is None:
        if isinstance(fill_value, bool):
            dtype = torch.bool
        elif isinstance(fill_value, int):
            dtype = torch.int64
        else:
            dtype = torch.get_default_dtype()
    else:
        fill_value = check_dtype(fill_value, dtype, device)

    # Extract shape from the 1D tensor
    shape = shape_tensor.tolist()

    # Handle empty shape
    if len(shape) == 0:
        shape = []

    out = torch.empty(shape, device=device, dtype=dtype)
    N = volume(shape)

    # If empty tensor, return
    if N == 0:
        return out

    # Dispatch to appropriate kernel based on fill_value type
    if isinstance(fill_value, torch.Tensor):
        return constant_of_shape_func(out, fill_value, out0=out)
    else:
        return constant_of_shape_func_scalar(out, fill_value, out0=out)