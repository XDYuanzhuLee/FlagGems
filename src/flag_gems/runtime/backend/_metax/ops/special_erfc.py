import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger("flag_gems." + __name__)
erfc = tl_extra_shim.erfc


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def special_erfc_func(x):
    output = erfc(x.to(tl.float32))
    return output


def erfc(x):
    logger.debug("METAX GEMS ERFC")
    return special_erfc_func(x)


def erfc_(x):
    logger.debug("METAX GEMS ERFC_")
    return special_erfc_func(x, out=x)