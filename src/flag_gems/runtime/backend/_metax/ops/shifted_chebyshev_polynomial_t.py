import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger("flag_gems." + __name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], is_tensor=[True, True])
@triton.jit
def shifted_chebyshev_polynomial_t_forward(x, n):
    """
    Compute shifted Chebyshev polynomial of the first kind T*_n(x).

    T*_n(x) = T_n(2x - 1), where T_n is the first kind Chebyshev polynomial.

    Uses the recurrence relation:
    T*_0(x) = 1
    T*_1(x) = 2x - 1
    T*_n(x) = 2(2x-1)T*_{n-1}(x) - T*_{n-2}(x)
    """
    n_int = n.to(tl.int32)
    x_2 = x.to(tl.float32) * 2.0 - 1.0

    # Base case n=0: T*_0(x) = 1
    result = tl.where(n_int == 0, tl.cast(1.0, tl.float32), tl.cast(0.0, tl.float32))

    # n=1: T*_1(x) = 2x - 1
    result = tl.where(n_int == 1, x_2, result)

    # n=2: T*_2(x) = 2*x_2^2 - 1
    t2 = 2.0 * x_2 * x_2 - 1.0
    result = tl.where(n_int == 2, t2, result)

    # n=3: T*_3(x) = 2*x_2*T*_2(x) - T*_1(x) = 2*x_2*t2 - x_2
    t3 = 2.0 * x_2 * t2 - x_2
    result = tl.where(n_int == 3, t3, result)

    # n=4: T*_4(x) = 2*x_2*T*_3(x) - T*_2(x)
    t4 = 2.0 * x_2 * t3 - t2
    result = tl.where(n_int == 4, t4, result)

    # n=5
    t5 = 2.0 * x_2 * t4 - t3
    result = tl.where(n_int == 5, t5, result)

    # n=6
    t6 = 2.0 * x_2 * t5 - t4
    result = tl.where(n_int == 6, t6, result)

    # n=7
    t7 = 2.0 * x_2 * t6 - t5
    result = tl.where(n_int == 7, t7, result)

    # n=8
    t8 = 2.0 * x_2 * t7 - t6
    result = tl.where(n_int == 8, t8, result)

    # n=9
    t9 = 2.0 * x_2 * t8 - t7
    result = tl.where(n_int == 9, t9, result)

    # n=10
    t10 = 2.0 * x_2 * t9 - t8
    result = tl.where(n_int == 10, t10, result)

    # n=11
    t11 = 2.0 * x_2 * t10 - t9
    result = tl.where(n_int == 11, t11, result)

    # n=12
    t12 = 2.0 * x_2 * t11 - t10
    result = tl.where(n_int == 12, t12, result)

    # n=13
    t13 = 2.0 * x_2 * t12 - t11
    result = tl.where(n_int == 13, t13, result)

    # n=14
    t14 = 2.0 * x_2 * t13 - t12
    result = tl.where(n_int == 14, t14, result)

    # n=15
    t15 = 2.0 * x_2 * t14 - t13
    result = tl.where(n_int == 15, t15, result)

    return result


def shifted_chebyshev_polynomial_t(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """
    Compute shifted Chebyshev polynomial of the first kind.

    Args:
        x: Input tensor.
        n: Degree of the polynomial (can be tensor or scalar).

    Returns:
        Tensor with the same shape as x.
    """
    logger.debug("METAX GEMS SHIFTED_CHEBYSHEV_POLYNOMIAL_T")

    # Handle scalar n
    if isinstance(n, int):
        n = torch.tensor(n, dtype=torch.int32, device=x.device)

    return shifted_chebyshev_polynomial_t_forward(x, n)