import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)

_acos = tl_extra_shim.acos


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def shifted_chebyshev_polynomial_t_func(x, n):
    # Shifted Chebyshev polynomial of the first kind T_n*(x)
    # T_n*(x) = T_n(2x - 1) = cos(n * acos(2x - 1))
    x_shifted = 2.0 * x - 1.0
    return tl.cos(n * _acos(x_shifted.to(tl.float32)))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def shifted_chebyshev_polynomial_t_func_tensor_scalar(x, n):
    x_shifted = 2.0 * x - 1.0
    return tl.cos(n * _acos(x_shifted.to(tl.float32)))


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def shifted_chebyshev_polynomial_t_func_scalar_tensor(x, n):
    x_shifted = 2.0 * x - 1.0
    return tl.cos(n * _acos(x_shifted.to(tl.float32)))


def shifted_chebyshev_polynomial_t(A, B):
    logger.debug("ILUVATAR GEMS SHIFTED_CHEBYSHEV_POLYNOMIAL_T")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return shifted_chebyshev_polynomial_t_func(A, B)
    elif isinstance(A, torch.Tensor):
        return shifted_chebyshev_polynomial_t_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return shifted_chebyshev_polynomial_t_func_scalar_tensor(A, B)
    else:
        # Both scalar
        x_shifted = 2.0 * A - 1.0
        import math
        return math.cos(B * math.acos(x_shifted))