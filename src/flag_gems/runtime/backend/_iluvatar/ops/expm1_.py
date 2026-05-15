import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)
expm1 = tl_extra_shim.expm1


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def expm1_func(x):
    return expm1(x)


def expm1_(A):
    logger.debug("ILUVATAR GEMS EXPM1_")
    expm1_func(A, out0=A)
    return A