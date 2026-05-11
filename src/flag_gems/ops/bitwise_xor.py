import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def ixor_func(x, y):
    return x ^ y


def bitwise_xor_tensor(A, B):
    logger.debug("GEMS BITWISE_XOR")
    return ixor_func(A, B)


def bitwise_xor_tensor_(A, B):
    logger.debug("GEMS BITWISE_XOR_")
    return ixor_func(A, B, out0=A)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def ixor_func_scalar(x, y):
    return x ^ y


def bitwise_xor_scalar(A, B):
    logger.debug("GEMS BITWISE_XOR SCALAR")
    return ixor_func_scalar(A, B)


def bitwise_xor_scalar_(A, B):
    logger.debug("GEMS BITWISE_XOR SCALAR_")
    return ixor_func_scalar(A, B, out0=A)


def bitwise_xor_scalar_tensor(A, B):
    logger.debug("GEMS BITWISE_XOR SCALAR TENSOR")
    return ixor_func_scalar(B, A)