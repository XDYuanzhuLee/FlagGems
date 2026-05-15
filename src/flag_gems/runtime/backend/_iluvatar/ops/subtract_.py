import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def sub_func(x, y, alpha):
    return x - y * alpha


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def sub_func_tensor_scalar(x, y, alpha):
    return x - y * alpha


def sub_(A, B, *, alpha=1):
    logger.debug("ILUVATAR GEMS SUB_")
    if isinstance(B, torch.Tensor):
        return sub_func(A, B, alpha, out0=A)
    else:
        return sub_func_tensor_scalar(A, B, alpha, out0=A)