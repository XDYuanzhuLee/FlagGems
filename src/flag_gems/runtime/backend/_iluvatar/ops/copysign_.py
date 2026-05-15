import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger("flag_gems." + __name__)


def _unwrap_if_constexpr(o):
    return o.value if isinstance(o, tl.constexpr) else o


@tl.constexpr
def _get_uint_dtype(num_bits):
    num_bits = _unwrap_if_constexpr(num_bits)
    return tl.core.get_int_dtype(num_bits, False)


@tl.constexpr
def _get_sign_bit_mask(num_bits):
    num_bits = _unwrap_if_constexpr(num_bits)
    return 1 << (num_bits - 1)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def copysign_func(input, other):
    abs_val = tl.abs(input)
    num_bits: tl.constexpr = input.dtype.primitive_bitwidth
    uint_dtype = _get_uint_dtype(num_bits)
    sign_bit_mask: tl.constexpr = _get_sign_bit_mask(num_bits)
    other_u = other.to(uint_dtype, bitcast=True)
    return tl.where((other_u & sign_bit_mask) != 0, -abs_val, abs_val)


def copysign_(input, other):
    logger.debug("ILUVATAR GEMS copysign_")
    return copysign_func(input, other)