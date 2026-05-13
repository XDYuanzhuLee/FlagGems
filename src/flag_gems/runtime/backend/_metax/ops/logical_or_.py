import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger("flag_gems." + __name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def logical_or_func(x, y):
    return x.to(tl.int1).logical_or(y.to(tl.int1))


def logical_or(A, B):
    logger.debug("METAX GEMS LOGICAL_OR")
    return logical_or_func(A, B)


def logical_or_(A, B):
    logger.debug("METAX GEMS LOGICAL_OR_")
    # Use out0 parameter for in-place operation
    logical_or_func(A, B, out0=A)
    return A