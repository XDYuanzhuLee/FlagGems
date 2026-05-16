import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def nextafter_func(x, y):
    # nextafter(x, y) returns y if x == y
    # Otherwise, returns the next representable value from x toward y

    # For float32 (most common): use int32 bitcast
    x_i32 = x.to(tl.int32, bitcast=True)
    direction = tl.where(y > x, tl.int32(1), tl.int32(-1))
    result_i32 = x_i32 + direction
    result = result_i32.to(tl.float32, bitcast=True)

    # Special case: if x == y, return y
    result = tl.where(x == y, y, result)

    return result


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def nextafter_func_tensor_scalar(x, y):
    x_i32 = x.to(tl.int32, bitcast=True)
    direction = tl.where(y > x, tl.int32(1), tl.int32(-1))
    result_i32 = x_i32 + direction
    result = result_i32.to(tl.float32, bitcast=True)
    result = tl.where(x == y, y, result)
    return result


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def nextafter_func_scalar_tensor(x, y):
    x_i32 = x.to(tl.int32, bitcast=True)
    direction = tl.where(y > x, tl.int32(1), tl.int32(-1))
    result_i32 = x_i32 + direction
    result = result_i32.to(tl.float32, bitcast=True)
    result = tl.where(x == y, y, result)
    return result


def nextafter_(A, B):
    logger.debug("ILUVATAR GEMS nextafter_")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return nextafter_func(A, B, out0=A)
    elif isinstance(A, torch.Tensor):
        return nextafter_func_tensor_scalar(A, B, out0=A)
    elif isinstance(B, torch.Tensor):
        return nextafter_func_scalar_tensor(A, B)
    else:
        # Both scalar
        import numpy as np
        return torch.tensor(np.nextafter(A, B))