import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger("flag_gems." + __name__)

# Get backend-agnostic math functions
cos = tl_extra_shim.cos
sin = tl_extra_shim.sin
acos = tl_extra_shim.acos


@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, 1, "INT_TO_FLOAT")],
)
@triton.jit
def chebyshev_polynomial_t_kernel(x, n):
    """
    Compute Chebyshev polynomial of the first kind T_n(x).

    If n = 0, returns 1.
    If n = 1, returns x.
    If n < 6 or |x| > 1, uses recursion: T_{n+1}(x) = 2*x*T_n(x) - T_{n-1}(x)
    Otherwise, uses explicit formula: T_n(x) = cos(n * arccos(x))
    """
    x_f32 = x.to(tl.float32)
    n_int = n.to(tl.int32)

    # Base cases
    # n = 0: T_0(x) = 1
    # n = 1: T_1(x) = x
    n_is_0 = n_int == 0
    n_is_1 = n_int == 1

    # Use recursion for small n or when |x| > 1
    use_recursion = (n_int < 6) | (tl.abs(x_f32) > 1.0)

    result: tl.float32

    if use_recursion:
        # Handle base cases first
        if n_is_0:
            result = 1.0
        elif n_is_1:
            result = x_f32
        else:
            # Recursion: T_{n+1}(x) = 2*x*T_n(x) - T_{n-1}(x)
            # T_0(x) = 1
            # T_1(x) = x
            t_prev = 1.0  # T_0
            t_curr = x_f32  # T_1
            # Compute from T_2 to T_n
            for i in range(2, 20):  # n won't exceed reasonable bounds
                # Check if we've reached n
                curr_is_n = i == n_int
                if curr_is_n:
                    break
                t_next = 2.0 * x_f32 * t_curr - t_prev
                t_prev = t_curr
                t_curr = t_next
            result = t_curr
    else:
        # Use explicit formula: T_n(x) = cos(n * arccos(x))
        result = cos(n_int.to(tl.float32) * acos(x_f32))

    # Combine results
    if n_is_0:
        result = 1.0
    elif n_is_1:
        result = x_f32

    return result.to(x.dtype)


def chebyshev_polynomial_t(x: torch.Tensor, n: torch.Tensor):
    """
    Compute Chebyshev polynomial of the first kind T_n(x).

    Args:
        x: Input tensor
        n: Degree of the polynomial (tensor, will be broadcast with x)

    Returns:
        Tensor with the same shape as broadcast(x, n)
    """
    logger.debug("METAX GEMS SPECIAL_CHEBYSHEV_POLYNOMIAL_T")
    return chebyshev_polynomial_t_kernel(x, n)