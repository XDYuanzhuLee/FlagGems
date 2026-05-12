import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger("flag_gems." + __name__)


@triton.jit
def _special_modified_bessel_i1_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # Compute in fp32 for accuracy/stability
    xf = x.to(tl.float32)
    ax = tl.abs(xf)

    # For I1(x), we use the following approximations:
    # Based on Numerical Recipes and Abramowitz & Stegun

    # Small region: x < 3.75
    # Use polynomial series expansion: I1(x) = sum_{k=0}^inf (x/2)^(2k+1) / (k! * (k+1)))
    t_small = ax / 3.75
    t2 = t_small * t_small

    # Polynomial coefficients for small region (Numerical Recipes)
    # I1(x) = x/2 * (1 + x^2/8 + x^4/96 + x^6/3072 + ...)
    p = 1.0 + t2 * (
        0.5
        + t2 * (
            0.125
            + t2 * (
                0.0625
                + t2 * (
                    0.0390625
                    + t2 * 0.026784896
                )
            )
        )
    )
    small = ax * p

    # Large region: x >= 3.75
    # Use asymptotic expansion for large x
    # I1(x) ~ exp(x) / sqrt(2*pi*x) * (1 - 3/(8x) + 15/(128x^2) - ...)
    t = 3.75 / ax
    q = 0.636619772 + t * (
        0.62164416
        + t * (
            0.21136030
            + t * (
                0.06235050
                + t * (
                    0.01077260
                    + t * 0.00149139
                )
            )
        )
    )
    large = q * tl.sqrt(ax) + q / (8.0 * ax) - q / (128.0 * ax * ax) * 3.0

    is_large = ax >= 3.75
    y = tl.where(is_large, large, small)

    # Handle negative input: I1(-x) = -I1(x)
    sign = tl.where(xf < 0, -1.0, 1.0)
    y = y * sign

    # Cast back to input dtype for storage
    y = y.to(x.dtype)
    tl.store(out_ptr + offsets, y, mask=mask)


def _run_special_modified_bessel_i1_kernel(x: torch.Tensor, out: torch.Tensor):
    assert x.is_cuda and out.is_cuda, "Tensors must be CUDA tensors"
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), "Unsupported dtype"
    assert out.dtype == x.dtype, "Output dtype must match input dtype"

    x_c = x.contiguous()
    out_c = out.contiguous()

    n_elements = out_c.numel()
    if n_elements == 0:
        return out

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _special_modified_bessel_i1_kernel[grid](x_c, out_c, n_elements, BLOCK_SIZE=1024)

    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out


def special_modified_bessel_i1(A: torch.Tensor) -> torch.Tensor:
    """
    Modified Bessel function of the first kind of order 1.
    ATen schema: aten::special_modified_bessel_i1(Tensor self) -> Tensor
    """
    logger.debug("METAX GEMS SPECIAL_MODIFIED_BESSEL_I1")
    out = torch.empty_like(A)
    return _run_special_modified_bessel_i1_kernel(A, out)