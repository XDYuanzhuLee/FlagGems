import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)
_acos = tl_extra_shim.acos


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit()
def acos_kernel(x):
    return _acos(x.to(tl.float32))


def acos_(A):
    logger.debug("ILUVATAR GEMS ACOS_")
    acos_kernel(A, out0=A)
    return A