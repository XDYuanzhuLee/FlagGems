import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, pointwise_dynamic
from flag_gems.utils import tl_extra_shim

logger = logging.getLogger("flag_gems." + __name__)

_pow = tl_extra_shim.pow


@pointwise_dynamic(promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def float_power_func(x, exponent):
    # float_power computes in double precision, but use float32 for metax compatibility
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def float_power_tensor_tensor(A, exponent):
    logger.debug("METAX GEMS FLOAT_POWER_TENSOR_TENSOR")
    return float_power_func(A, exponent, out0=A)


def float_power_tensor_tensor_(A, exponent):
    logger.debug("METAX GEMS FLOAT_POWER_TENSOR_TENSOR_")
    return float_power_func(A, exponent, out0=A)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def float_power_func_scalar(x, exponent):
    # float_power computes in double precision, but use float32 for metax compatibility
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def float_power_tensor_scalar(A, exponent):
    logger.debug("METAX GEMS FLOAT_POWER_TENSOR_SCALAR")
    return float_power_func_scalar(A, exponent)


def float_power_tensor_scalar_(A, exponent):
    logger.debug("METAX GEMS FLOAT_POWER_TENSOR_SCALAR_")
    return float_power_func_scalar(A, exponent, out0=A)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def float_power_func_scalar_tensor(x, exponent):
    # float_power computes in double precision, but use float32 for metax compatibility
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def float_power_scalar(A, exponent):
    logger.debug("METAX GEMS FLOAT_POWER_SCALAR")
    return float_power_func_scalar_tensor(A, exponent)