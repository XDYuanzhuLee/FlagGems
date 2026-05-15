import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)

erfinv = tl_extra_shim.erfinv


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def special_ndtri_func(x):
    # ndtri(p) = sqrt(2) * erfinv(2p - 1)
    # where erfinv is the inverse of the error function
    # Input p should be in range [0, 1]
    two = 2.0
    sqrt_two = 1.4142135623730951  # sqrt(2)
    one = 1.0

    # Transform p to the erfinv input domain: 2p - 1
    # This maps [0, 1] to [-1, 1]
    y = two * x - one

    # Compute erfinv(y)
    result = erfinv(y)

    # Multiply by sqrt(2) to get ndtri
    result = result * sqrt_two

    return result


def special_ndtri(x: torch.Tensor):
    logger.debug("ILUVATAR GEMS special_ndtri")
    return special_ndtri_func(x)


def special_ndtri_out(x: torch.Tensor, out: torch.Tensor):
    logger.debug("ILUVATAR GEMS special_ndtri_out")
    return special_ndtri_func(x, out0=out)