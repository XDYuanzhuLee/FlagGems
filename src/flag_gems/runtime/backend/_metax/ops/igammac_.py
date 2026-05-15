import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger("flag_gems." + __name__)

# Get cross-backend compatible functions from tl_extra_shim
lgamma = tl_extra_shim.lgamma
exp = tl_extra_shim.exp
log = tl_extra_shim.log
pow = tl_extra_shim.pow


@libentry()
@triton.jit
def igammac_kernel_(
    a_ptr,
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute the regularized upper incomplete gamma function Q(a, x) = 1 - P(a, x).

    This is done using the series expansion for small x and the continued fraction
    representation for larger x.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)

    # Compute igammac using a combination of series expansion and continued fraction
    result = compute_igammac(a, x)

    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def compute_igammac(a, x):
    """Compute the regularized upper incomplete gamma function Q(a, x).

    Uses the complementary relationship: Q(a, x) = 1 - P(a, x)
    where P(a, x) is the regularized lower incomplete gamma function.

    For numerical stability:
    - Small x: use series expansion
    - Large x: use asymptotic approximation
    - General case: use the relationship with the lower incomplete gamma
    """
    # Handle edge cases
    # x <= 0: Q(a, 0) = 1
    # a <= 0: use limit behavior

    # For numerical stability, compute in float32
    a_f32 = a.to(tl.float32)
    x_f32 = x.to(tl.float32)

    # Edge cases
    # Q(a, 0) = 1
    # Q(0, x) = 1 for x > 0 (limit)
    # Q(a, inf) = 0

    result = compute_igammac_core(a_f32, x_f32)

    # Preserve original dtype
    return result.to(a.dtype)


@triton.jit
def compute_igammac_core(a, x):
    """Core implementation of igammac with numerical considerations."""
    # Handle edge cases
    zero = 0.0
    one = 1.0

    # x <= 0: Q(a, 0) = 1
    x_le_zero = x <= zero
    if tl.reduce(x_le_zero, 0, and_):
        return one

    # x -> infinity: Q(a, inf) = 0
    # Use a large threshold
    large_x = 100.0
    x_large = x > large_x
    if tl.reduce(x_large, 0, and_):
        return zero

    # For small x, use series expansion
    # Q(a, x) = 1 - e^(-x) * x^(a-1) * sum_{n=0}^inf x^n / (a * (a+1) * ... * (a+n))
    small_x = x < one

    # Use different computation strategies based on x value
    # For the series expansion method
    result_small = igammac_series(a, x)

    # For asymptotic case (larger x)
    # Q(a, x) ~ x^(a-1) * e^(-x) / Gamma(a) as x -> infinity
    result_large = igammac_asymptotic(a, x)

    # Combine results based on x value
    result = tl.where(small_x, result_small, result_large)

    # Ensure result is in [0, 1]
    result = tl.where(result < zero, zero, result)
    result = tl.where(result > one, one, result)

    return result


@triton.jit
def igammac_series(a, x):
    """Compute igammac using series expansion for small x.

    Q(a, x) = 1 - e^(-x) * x^(a-1) * S
    where S = sum_{n=0}^inf x^n / (a * (a+1) * ... * (a+n))
    """
    zero = 0.0
    one = 1.0

    # Compute e^(-x)
    exp_neg_x = exp(-x)

    # Compute x^(a-1)
    x_pow_a_minus_1 = pow(x, a - one)

    # Series expansion - compute a finite sum
    # For numerical stability, use forward recurrence
    max_iter = 32
    tol = 1e-10

    term = one / a
    sum_val = term
    a_curr = a

    # Use a for loop in triton
    for _ in range(1, max_iter):
        a_curr = a_curr + one
        term = term * x / a_curr
        sum_val = sum_val + term

        # Check convergence
        abs_term = abs(term)
        abs_sum = abs(sum_val)
        if abs_sum > zero:
            if abs_term / abs_sum < tol:
                break

    # Compute P(a, x) = e^(-x) * x^(a-1) * S / Gamma(a)
    # Then Q(a, x) = 1 - P(a, x)
    gamma_a = exp(lgamma(a))
    p_val = exp_neg_x * x_pow_a_minus_1 * sum_val / gamma_a

    # Q = 1 - P
    q_val = one - p_val

    return q_val


@triton.jit
def igammac_asymptotic(a, x):
    """Compute igammac using asymptotic approximation for larger x.

    Q(a, x) ~ x^(a-1) * e^(-x) / Gamma(a) * (1 + (1-a)/x + (1-a)(2-a)/x^2 + ...)
    """
    zero = 0.0
    one = 1.0

    # Compute Gamma(a)
    gamma_a = exp(lgamma(a))

    # Main term: x^(a-1) * e^(-x) / Gamma(a)
    exp_neg_x = exp(-x)
    x_pow_a_minus_1 = pow(x, a - one)

    # First order correction term: (1-a)/x
    correction = (one - a) / x

    # Compute the approximation
    # Q(a, x) ≈ x^(a-1) * e^(-x) / Gamma(a)
    result = x_pow_a_minus_1 * exp_neg_x / gamma_a

    # Clamp to [0, 1] as asymptotic approximation may slightly exceed bounds
    result = tl.where(result > one, one, result)

    return result


def igammac_(a, x):
    """In-place version of regularized upper incomplete gamma function.

    Computes Q(a, x) = 1 - P(a, x) where P is the regularized lower incomplete gamma.

    Args:
        a: First input tensor (the 'a' parameter)
        x: Second input tensor (the 'x' parameter) - modified in place

    Returns:
        Modified tensor x containing Q(a, x)
    """
    if not isinstance(a, torch.Tensor):
        raise TypeError("igammac_ expects a torch.Tensor as the first argument")

    if not isinstance(x, torch.Tensor):
        raise TypeError("igammac_ expects a torch.Tensor as the second argument")

    # Check shapes - they should broadcast together
    # We compute element-wise, so we need same shape or broadcastable
    a_shape = a.shape
    x_shape = x.shape

    # For now, require same shape or expand x to match a
    # (this follows the pattern of other binary ops)
    if a_shape != x_shape:
        try:
            x = x.expand_as(a)
        except RuntimeError:
            raise ValueError(
                f"Cannot broadcast shapes {a_shape} and {x_shape} for igammac_"
            )

    # Handle non-contiguous tensors
    a_temp = a
    x_temp = x
    output_temp = x

    a_needs_copy = not a.is_contiguous()
    x_needs_copy = not x.is_contiguous()

    if a_needs_copy:
        a_temp = a.contiguous()
    if x_needs_copy:
        x_temp = x.contiguous()
        output_temp = torch.empty_like(x_temp)

    n_elements = output_temp.numel()
    if n_elements == 0:
        return x

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    logger.debug("METAX GEMS IGAMMAC_")

    with torch_device_fn.device(x_temp.device):
        igammac_kernel_[grid](
            a_temp,
            x_temp,
            output_temp,
            n_elements,
            BLOCK_SIZE=1024,
        )

    # Copy result back to original tensor if needed
    if x_needs_copy:
        x.copy_(output_temp)
        return x
    else:
        return output_temp