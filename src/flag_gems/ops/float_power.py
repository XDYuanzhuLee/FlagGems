import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_pow = tl_extra_shim.pow
logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def float_power_func(x, exponent):
    # float_power computes in double precision (float64)
    return _pow(x.to(tl.float64), exponent.to(tl.float64))


def float_power_tensor_tensor(A, exponent):
    logger.debug("GEMS FLOAT_POWER_TENSOR_TENSOR")
    return float_power_func(A, exponent)


def float_power_tensor_tensor_(A, exponent):
    logger.debug("GEMS FLOAT_POWER_TENSOR_TENSOR_")
    return float_power_func(A, exponent, out0=A)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def float_power_func_tensor_scalar(x, exponent):
    # float_power computes in double precision (float64)
    return _pow(x.to(tl.float64), exponent.to(tl.float64))


def float_power_tensor_scalar(A, exponent):
    logger.debug("GEMS FLOAT_POWER_TENSOR_SCALAR")
    return float_power_func_tensor_scalar(A, exponent)


def float_power_tensor_scalar_(A, exponent):
    logger.debug("GEMS FLOAT_POWER_TENSOR_SCALAR_")
    return float_power_func_tensor_scalar(A, exponent, out0=A)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def float_power_func_scalar_tensor(x, exponent):
    # float_power computes in double precision (float64)
    return _pow(x.to(tl.float64), exponent.to(tl.float64))


def float_power_scalar(A, exponent):
    logger.debug("GEMS FLOAT_POWER_SCALAR")
    return float_power_func_scalar_tensor(A, exponent)