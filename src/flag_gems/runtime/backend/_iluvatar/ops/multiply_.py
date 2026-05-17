import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def mul_func(x, y):
    return x * y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def mul_func_scalar(x, y):
    return x * y


def multiply_(A, B):
    logger.debug("ILUVATAR GEMS MULTIPLY_")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return mul_func(A, B, out0=A)
    elif isinstance(A, torch.Tensor):
        return mul_func_scalar(A, B, out0=A)
    else:
        raise ValueError("multiply_ requires the first argument to be a tensor")


# mul_ is the primary name used in flag_gems, multiply_ is an alias
mul_ = multiply_