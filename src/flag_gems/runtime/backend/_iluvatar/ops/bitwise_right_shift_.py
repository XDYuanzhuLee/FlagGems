import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_right_shift_kernel(a, b):
    return a >> b


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_right_shift_kernel_tensor_scalar(a, b):
    return a >> b


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_right_shift_kernel_scalar_tensor(a, b):
    return a >> b


def bitwise_right_shift_(self, other):
    logger.debug("ILUVATAR GEMS BITWISE_RIGHT_SHIFT_")
    if isinstance(self, torch.Tensor) and isinstance(other, torch.Tensor):
        return bitwise_right_shift_kernel(self, other, out0=self)
    elif isinstance(self, torch.Tensor):
        return bitwise_right_shift_kernel_tensor_scalar(self, other, out0=self)
    else:
        # Both scalar - return a new tensor
        return torch.tensor(self >> other)