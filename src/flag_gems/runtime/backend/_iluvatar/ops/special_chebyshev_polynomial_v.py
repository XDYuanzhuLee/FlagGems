import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger("flag_gems." + __name__)


@triton.jit
def _chebyshev_polynomial_v_kernel(x, n, out, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < tl.numel(out)

    x_val = tl.load(x + offsets, mask=mask)
    n_val = tl.load(n + offsets, mask=mask)

    # V_0(x) = 1
    # V_1(x) = 2*x + 1
    # V_n(x) = 2*x*V_{n-1}(x) - V_{n-2}(x)

    # Use recurrence relation to compute Chebyshev polynomial V_n(x)
    # For small n, we can compute directly
    # For efficiency, we use the iterative approach

    n_int = n_val.to(tl.int32)

    # Compute in fp32 for accuracy
    x_fp32 = x_val.to(tl.float32)

    one = tl.cast(1.0, tl.float32)
    two = tl.cast(2.0, tl.float32)

    # v0 = V_0(x) = 1
    v_prev2 = one
    # v1 = V_1(x) = 2*x + 1
    v_prev1 = two * x_fp32 + one

    # For n == 0, return 1
    # For n == 1, return 2*x + 1
    result = tl.where(n_int == 0, one, tl.where(n_int == 1, v_prev1, one))

    # Iterate for n >= 2
    # Use a loop for small n, for larger n we need a more efficient approach
    # Chebyshev polynomials can be computed using:
    # V_n(x) = sin((n+1)*arccos(x)) / sin(arccos(x))
    # But this has numerical issues for large n

    # Use the recurrence relation: V_n(x) = 2*x*V_{n-1}(x) - V_{n-2}(x)
    for i in range(2, 128):  # Max n we support in the loop
        # Only compute if n >= i
        cond = n_int >= i
        v_curr = two * x_fp32 * v_prev1 - v_prev2
        v_curr = tl.where(cond, v_curr, result)
        result = tl.where(cond, v_curr, result)
        v_prev2 = tl.where(cond, v_prev1, v_prev2)
        v_prev1 = tl.where(cond, v_curr, v_prev1)

    # Convert back to input dtype
    result = result.to(x_val.dtype)
    tl.store(out + offsets, result, mask=mask)


def _run_chebyshev_polynomial_v_kernel(x, n, out):
    assert x.is_cuda and n.is_cuda and out.is_cuda, "Tensors must be CUDA tensors"
    assert x.shape == n.shape, "Input shapes must match"
    assert x.shape == out.shape, "Output shape must match input shape"
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), "Unsupported dtype"
    assert n.dtype in (
        torch.int32,
        torch.int64,
    ), "n must be integer type"

    x_c = x.contiguous()
    n_c = n.contiguous()
    out_c = out.contiguous()

    n_elements = out_c.numel()
    if n_elements == 0:
        return out

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _chebyshev_polynomial_v_kernel[grid](x_c, n_c, out_c, BLOCK_SIZE=1024)

    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def chebyshev_polynomial_v_func(x, n):
    # V_0(x) = 1
    # V_1(x) = 2*x + 1
    # V_n(x) = 2*x*V_{n-1}(x) - V_{n-2}(x)
    # This kernel computes the polynomial using the recurrence relation

    n_int = n.to(tl.int32)

    one = tl.cast(1.0, x.dtype)
    two = tl.cast(2.0, x.dtype)

    # Convert x to float for computation
    x_fp = x.to(tl.float32)

    # V_0(x) = 1
    v_prev2 = one
    # V_1(x) = 2*x + 1
    v_prev1 = two * x_fp + one

    # Initialize result
    result = tl.where(n_int == 0, one, tl.where(n_int == 1, v_prev1, one))

    # Iteratively compute for n >= 2
    # Use loop unrolling for efficiency
    for i in range(2, 64):
        cond = n_int >= i
        v_curr = two * x_fp * v_prev1 - v_prev2
        result = tl.where(cond, v_curr, result)
        v_prev2 = tl.where(cond, v_prev1, v_prev2)
        v_prev1 = tl.where(cond, v_curr, v_prev1)

    # Handle n >= 64 using the same loop but with different max
    for i in range(64, 128):
        cond = n_int >= i
        v_curr = two * x_fp * v_prev1 - v_prev2
        result = tl.where(cond, v_curr, result)
        v_prev2 = tl.where(cond, v_prev1, v_prev2)
        v_prev1 = tl.where(cond, v_curr, v_prev1)

    return result.to(x.dtype)


def chebyshev_polynomial_v(x, n):
    """Compute Chebyshev polynomial of the third kind V_n(x).

    Args:
        x: Input tensor
        n: Degree of the polynomial (int or tensor)

    Returns:
        Tensor with the same shape as x
    """
    logger.debug("ILUVATAR GEMS SPECIAL_CHEBYSHEV_POLYNOMIAL_V")

    # Handle n as scalar
    if isinstance(n, int):
        n = torch.tensor(n, dtype=torch.int32, device=x.device)

    # Broadcast n to match x shape if needed
    if n.shape != x.shape:
        n = n.expand(x.shape)

    # Create output tensor
    out = torch.empty_like(x)

    return chebyshev_polynomial_v_func(x, n, out=out)


def chebyshev_polynomial_v_out(x, n, out):
    """Compute Chebyshev polynomial of the third kind V_n(x) with output tensor.

    Args:
        x: Input tensor
        n: Degree of the polynomial (int or tensor)
        out: Output tensor

    Returns:
        Output tensor
    """
    logger.debug("ILUVATAR GEMS SPECIAL_CHEBYSHEV_POLYNOMIAL_V_OUT")

    # Handle n as scalar
    if isinstance(n, int):
        n = torch.tensor(n, dtype=torch.int32, device=x.device)

    # Broadcast n to match x shape if needed
    if n.shape != x.shape:
        n = n.expand(x.shape)

    return chebyshev_polynomial_v_func(x, n, out=out)