import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def chebyshev_polynomial_t_func(x, n):
    # Chebyshev polynomial of the first kind T_n(x)
    # Using explicit formulas for n = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

    n_int = n.to(tl.int32)
    zero = tl.zeros_like(x)
    one = zero + 1.0

    # Base cases
    result = tl.where(n_int == 0, one, tl.zeros_like(x))
    result = tl.where(n_int == 1, x, result)

    # T_2(x) = 2*x^2 - 1
    cond_2 = n_int == 2
    t_2 = 2.0 * x * x - 1.0
    result = tl.where(cond_2, t_2, result)

    # T_3(x) = 4*x^3 - 3*x
    cond_3 = n_int == 3
    t_3 = (4.0 * x * x - 3.0) * x
    result = tl.where(cond_3, t_3, result)

    # T_4(x) = 8*x^4 - 8*x^2 + 1
    cond_4 = n_int == 4
    x_sq = x * x
    t_4 = 8.0 * x_sq * x_sq - 8.0 * x_sq + 1.0
    result = tl.where(cond_4, t_4, result)

    # T_5(x) = 16*x^5 - 20*x^3 + 5*x
    cond_5 = n_int == 5
    t_5 = (16.0 * x_sq * x - 20.0 * x) * x_sq + 5.0 * x
    result = tl.where(cond_5, t_5, result)

    # For n >= 6, use iterative recurrence or explicit formulas
    # T_6(x) = 32*x^6 - 48*x^4 + 18*x^2 - 1
    cond_6 = n_int == 6
    x_4 = x_sq * x_sq
    t_6 = 32.0 * x_4 * x_sq - 48.0 * x_4 + 18.0 * x_sq - 1.0
    result = tl.where(cond_6, t_6, result)

    # T_7(x) = 64*x^7 - 112*x^5 + 56*x^3 - 7*x
    cond_7 = n_int == 7
    x_6 = x_4 * x_sq
    t_7 = 64.0 * x_6 - 112.0 * x_4 * x + 56.0 * x_sq * x - 7.0 * x
    result = tl.where(cond_7, t_7, result)

    # T_8(x) = 128*x^8 - 256*x^6 + 160*x^4 - 32*x^2 + 1
    cond_8 = n_int == 8
    x_8 = x_4 * x_4
    t_8 = 128.0 * x_8 - 256.0 * x_6 + 160.0 * x_4 - 32.0 * x_sq + 1.0
    result = tl.where(cond_8, t_8, result)

    # T_9(x) = 256*x^9 - 576*x^7 + 432*x^5 - 120*x^3 + 9*x
    cond_9 = n_int == 9
    x_8_sq = x_8 * x
    t_9 = 256.0 * x_8_sq - 576.0 * x_6 * x + 432.0 * x_4 * x - 120.0 * x_sq * x + 9.0 * x
    result = tl.where(cond_9, t_9, result)

    # T_10(x) = 512*x^10 - 1280*x^8 + 1120*x^6 - 400*x^4 + 50*x^2 - 1
    cond_10 = n_int == 10
    x_10 = x_8 * x_sq
    t_10 = 512.0 * x_10 - 1280.0 * x_8 + 1120.0 * x_6 - 400.0 * x_4 + 50.0 * x_sq - 1.0
    result = tl.where(cond_10, t_10, result)

    return result


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def chebyshev_polynomial_t_func_ts(x, n):
    # Same implementation for tensor-scalar case
    n_int = n.to(tl.int32)
    zero = tl.zeros_like(x)
    one = zero + 1.0

    result = tl.where(n_int == 0, one, tl.zeros_like(x))
    result = tl.where(n_int == 1, x, result)

    cond_2 = n_int == 2
    t_2 = 2.0 * x * x - 1.0
    result = tl.where(cond_2, t_2, result)

    cond_3 = n_int == 3
    t_3 = (4.0 * x * x - 3.0) * x
    result = tl.where(cond_3, t_3, result)

    cond_4 = n_int == 4
    x_sq = x * x
    t_4 = 8.0 * x_sq * x_sq - 8.0 * x_sq + 1.0
    result = tl.where(cond_4, t_4, result)

    cond_5 = n_int == 5
    t_5 = (16.0 * x_sq * x - 20.0 * x) * x_sq + 5.0 * x
    result = tl.where(cond_5, t_5, result)

    cond_6 = n_int == 6
    x_4 = x_sq * x_sq
    t_6 = 32.0 * x_4 * x_sq - 48.0 * x_4 + 18.0 * x_sq - 1.0
    result = tl.where(cond_6, t_6, result)

    cond_7 = n_int == 7
    x_6 = x_4 * x_sq
    t_7 = 64.0 * x_6 - 112.0 * x_4 * x + 56.0 * x_sq * x - 7.0 * x
    result = tl.where(cond_7, t_7, result)

    cond_8 = n_int == 8
    x_8 = x_4 * x_4
    t_8 = 128.0 * x_8 - 256.0 * x_6 + 160.0 * x_4 - 32.0 * x_sq + 1.0
    result = tl.where(cond_8, t_8, result)

    cond_9 = n_int == 9
    x_8_sq = x_8 * x
    t_9 = 256.0 * x_8_sq - 576.0 * x_6 * x + 432.0 * x_4 * x - 120.0 * x_sq * x + 9.0 * x
    result = tl.where(cond_9, t_9, result)

    cond_10 = n_int == 10
    x_10 = x_8 * x_sq
    t_10 = 512.0 * x_10 - 1280.0 * x_8 + 1120.0 * x_6 - 400.0 * x_4 + 50.0 * x_sq - 1.0
    result = tl.where(cond_10, t_10, result)

    return result


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def chebyshev_polynomial_t_func_st(x, n):
    # Same implementation for scalar-tensor case
    n_int = n.to(tl.int32)
    zero = tl.zeros_like(x)
    one = zero + 1.0

    result = tl.where(n_int == 0, one, tl.zeros_like(x))
    result = tl.where(n_int == 1, x, result)

    cond_2 = n_int == 2
    t_2 = 2.0 * x * x - 1.0
    result = tl.where(cond_2, t_2, result)

    cond_3 = n_int == 3
    t_3 = (4.0 * x * x - 3.0) * x
    result = tl.where(cond_3, t_3, result)

    cond_4 = n_int == 4
    x_sq = x * x
    t_4 = 8.0 * x_sq * x_sq - 8.0 * x_sq + 1.0
    result = tl.where(cond_4, t_4, result)

    cond_5 = n_int == 5
    t_5 = (16.0 * x_sq * x - 20.0 * x) * x_sq + 5.0 * x
    result = tl.where(cond_5, t_5, result)

    cond_6 = n_int == 6
    x_4 = x_sq * x_sq
    t_6 = 32.0 * x_4 * x_sq - 48.0 * x_4 + 18.0 * x_sq - 1.0
    result = tl.where(cond_6, t_6, result)

    cond_7 = n_int == 7
    x_6 = x_4 * x_sq
    t_7 = 64.0 * x_6 - 112.0 * x_4 * x + 56.0 * x_sq * x - 7.0 * x
    result = tl.where(cond_7, t_7, result)

    cond_8 = n_int == 8
    x_8 = x_4 * x_4
    t_8 = 128.0 * x_8 - 256.0 * x_6 + 160.0 * x_4 - 32.0 * x_sq + 1.0
    result = tl.where(cond_8, t_8, result)

    cond_9 = n_int == 9
    x_8_sq = x_8 * x
    t_9 = 256.0 * x_8_sq - 576.0 * x_6 * x + 432.0 * x_4 * x - 120.0 * x_sq * x + 9.0 * x
    result = tl.where(cond_9, t_9, result)

    cond_10 = n_int == 10
    x_10 = x_8 * x_sq
    t_10 = 512.0 * x_10 - 1280.0 * x_8 + 1120.0 * x_6 - 400.0 * x_4 + 50.0 * x_sq - 1.0
    result = tl.where(cond_10, t_10, result)

    return result


def special_chebyshev_polynomial_t(x, n):
    logger.debug("ILUVATAR GEMS special_chebyshev_polynomial_t")
    if isinstance(x, torch.Tensor) and isinstance(n, torch.Tensor):
        return chebyshev_polynomial_t_func(x, n)
    elif isinstance(x, torch.Tensor):
        return chebyshev_polynomial_t_func_ts(x, n)
    elif isinstance(n, torch.Tensor):
        return chebyshev_polynomial_t_func_st(x, n)
    else:
        return torch.tensor(torch.special.chebyshev_polynomial_t(x, n))