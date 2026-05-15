import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def cosh_kernel(x):
    # cosh(x) = (e^x + e^(-x)) / 2
    # Use float32 for intermediate computation to improve accuracy
    x = x.to(tl.float32)
    e_pos = tl.exp(x)
    e_neg = tl.exp(-x)
    return 0.5 * (e_pos + e_neg)


def cosh(A):
    logger.debug("GEMS COSH")
    return cosh_kernel(A)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def cosh_inplace_kernel(x):
    # cosh(x) = (e^x + e^(-x)) / 2
    x = x.to(tl.float32)
    e_pos = tl.exp(x)
    e_neg = tl.exp(-x)
    return 0.5 * (e_pos + e_neg)


def cosh_(A):
    logger.debug("GEMS COSH_")
    return cosh_inplace_kernel(A, out0=A)