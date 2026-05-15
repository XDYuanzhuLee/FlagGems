import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger("flag_gems." + __name__)

# Get bessel functions from tl_extra_shim (cross-backend compatible)
_yn = tl_extra_shim.yn


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def special_bessel_y1_forward(x):
    # y1(x) = yn(1, x) - Bessel function of the second kind, order 1
    return _yn(1, x.to(tl.float32))


def special_bessel_y1(A):
    """Compute the Bessel function of the second kind of order 1."""
    logger.debug("METAX GEMS SPECIAL_BESSEL_Y1")
    return special_bessel_y1_forward(A)