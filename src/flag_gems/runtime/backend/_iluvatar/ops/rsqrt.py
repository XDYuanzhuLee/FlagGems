import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger("flag_gems." + __name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def rsqrt_func(x):
    # Convert to float32 for compatibility with iluvatar backend
    # then convert back to the original type
    x_fp32 = x.to(tl.float32)
    result = 1.0 / tl.sqrt(x_fp32)
    return result.to(x.type)


def rsqrt(A):
    logger.debug("ILUVATAR GEMS RSQRT")
    return rsqrt_func(A)


def rsqrt_(A):
    logger.debug("ILUVATAR GEMS RSQRT_")
    return rsqrt_func(A, out0=A)