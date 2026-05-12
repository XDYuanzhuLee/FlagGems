import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger("flag_gems." + __name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_right_shift_kernel(a, b):
    return a >> b


def bitwise_right_shift(self, other, *, out=None):
    logger.debug("METAX GEMS BITWISE_RIGHT_SHIFT")
    return bitwise_right_shift_kernel(self, other, out=out)