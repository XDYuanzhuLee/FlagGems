import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def xlogy_tensor_tensor(x, y):
    # xlogy(x, y) = x * log(y)
    # Special cases:
    # - If x == 0, return 0 (even if y is NaN or negative)
    # - If y is NaN, return NaN (unless x == 0)
    # - Otherwise, return x * log(y)
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    # Check if x is zero
    is_zero = x_f32 == 0.0
    # Compute log(y) - this will be NaN if y <= 0
    log_y = tl.log(y_f32)
    # Compute x * log(y)
    result = x_f32 * log_y
    # Return 0 where x is 0, otherwise return result (which handles NaN case)
    return tl.where(is_zero, 0.0, result)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def xlogy_tensor_scalar(x, y):
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    is_zero = x_f32 == 0.0
    log_y = tl.log(y_f32)
    result = x_f32 * log_y
    return tl.where(is_zero, 0.0, result)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def xlogy_scalar_tensor(x, y):
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    is_zero = x_f32 == 0.0
    log_y = tl.log(y_f32)
    result = x_f32 * log_y
    return tl.where(is_zero, 0.0, result)


def xlogy(A, B):
    logger.debug("ILUVATAR GEMS XLOGY")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return xlogy_tensor_tensor(A, B)
    elif isinstance(A, torch.Tensor):
        return xlogy_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return xlogy_scalar_tensor(A, B)
    else:
        # Both scalar
        if A == 0:
            return torch.tensor(0.0)
        return torch.tensor(A * torch.log(B).item())