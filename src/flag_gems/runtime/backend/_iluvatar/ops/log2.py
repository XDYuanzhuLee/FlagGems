import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def log2_func(x):
    # log2(x) = log(x) / log(2)
    return tl.log(x.to(tl.float32)) / tl.log(2.0)


def log2(A):
    logger.debug("GEMS LOG2")
    return log2_func(A)


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def log2_func_(x):
    # log2(x) = log(x) / log(2)
    return tl.log(x.to(tl.float32)) / tl.log(2.0)


def log2_(A):
    logger.debug("GEMS LOG2_")
    return log2_func_(A, out0=A)