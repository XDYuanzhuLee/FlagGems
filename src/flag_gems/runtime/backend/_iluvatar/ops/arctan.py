import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger("flag_gems." + __name__)
_atan = tl_extra_shim.atan


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def arctan_kernel(x):
    return _atan(x.to(tl.float32))


def arctan(A):
    logger.debug("ILUVATAR GEMS ARCTAN")
    out = arctan_kernel(A)
    return out


def arctan_(A):
    logger.debug("ILUVATAR GEMS ARCTAN_")
    arctan_kernel(A, out0=A)
    return A


# atan is an alias for arctan in PyTorch, so we export both names
atan = arctan
atan_ = arctan_