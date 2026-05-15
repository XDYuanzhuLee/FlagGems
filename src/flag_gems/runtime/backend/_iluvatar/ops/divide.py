import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)
div_rn = tl_extra_shim.div_rn
div_rz = tl_extra_shim.div_rz
fmod = tl_extra_shim.fmod
trunc = tl_extra_shim.trunc


@pointwise_dynamic(promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def divide_func(x, y):
    return x / y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def divide_func_tensor_scalar(x, y):
    return x / y


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "INT_TO_FLOAT")])
@triton.jit
def divide_func_scalar_tensor(x, y):
    return x / y


def divide(A, B):
    logger.debug("ILUVATAR GEMS DIVIDE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return divide_func(A, B)
    elif isinstance(A, torch.Tensor):
        return divide_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return divide_func_scalar_tensor(A, B)
    else:
        return torch.tensor(A / B)


def divide_(A, B):
    logger.debug("ILUVATAR GEMS DIVIDE_")
    if isinstance(B, torch.Tensor):
        return divide_func(A, B, out0=A)
    else:
        return divide_func_tensor_scalar(A, B, out0=A)