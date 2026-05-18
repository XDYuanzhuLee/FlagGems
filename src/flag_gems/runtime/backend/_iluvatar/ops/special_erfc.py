import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger("flag_gems." + __name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def erfc_func(x):
    # erfc(x) = 1 - erf(x)
    return 1.0 - tl.math.erf(x.to(tl.float32))


def special_erfc(A):
    logger.debug("ILUVATAR GEMS SPECIAL_ERFC")
    return erfc_func(A)


def special_erfc_(A):
    logger.debug("ILUVATAR GEMS SPECIAL_ERFC_")
    return erfc_func(A, out0=A)


def special_erfc_out(A, out):
    logger.debug("ILUVATAR GEMS SPECIAL_ERFC_OUT")
    return erfc_func(A, out0=out)