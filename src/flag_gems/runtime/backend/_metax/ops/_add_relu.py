import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger("flag_gems." + __name__)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def add_relu_func(x, y, alpha):
    # Compute: relu(x + alpha * y)
    result = x + y * alpha
    return tl.where(result > 0, result, 0.0)


@pointwise_dynamic(is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def add_relu_func_tensor_scalar(x, y, alpha):
    # Compute: relu(x + alpha * y)
    result = x + y * alpha
    return tl.where(result > 0, result, 0.0)


@pointwise_dynamic(is_tensor=[False, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def add_relu_func_scalar_tensor(x, y, alpha):
    # Compute: relu(x + alpha * y)
    result = x + y * alpha
    return tl.where(result > 0, result, 0.0)


def add_relu(A, B, *, alpha=1):
    logger.debug("METAX GEMS ADD_RELU")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        return add_relu_func(A, B, alpha)
    elif isinstance(A, torch.Tensor):
        return add_relu_func_tensor_scalar(A, B, alpha)
    elif isinstance(B, torch.Tensor):
        return add_relu_func_scalar_tensor(A, B, alpha)
    else:
        return torch.tensor(max(0, A + B * alpha))