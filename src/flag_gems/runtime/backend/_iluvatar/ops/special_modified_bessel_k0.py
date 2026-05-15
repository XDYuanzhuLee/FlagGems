# Generated for Iluvatar (天数) backend
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@triton.jit
def special_modified_bessel_k0_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    ax = tl.abs(x_f32)

    # Special case: x = 0, K0(0) = +inf
    # Based on Cephes library implementation

    # Small x region: 0 < x <= 2
    # K0(x) = -ln(x/2) * I0(x) + poly(x)
    y = x_f32 * x_f32 / 4.0

    # Polynomial for K0 series (Cephes)
    p0 = -0.57721566
    p1 = 0.42278434
    p2 = 0.23069756
    p3 = 0.0348859
    p4 = 0.00262698
    p5 = 0.0001075
    p6 = 0.0007414e-5

    p_series = p0 + y * (p1 + y * (p2 + y * (p3 + y * (p4 + y * (p5 + y * p6)))))

    # I0(x) polynomial for small x (|x| < 3.75)
    y_i0 = x_f32 / 3.75
    y_i0_2 = y_i0 * y_i0
    i0 = 1.0 + y_i0_2 * (3.5156229 + y_i0_2 * (3.0899424 + y_i0_2 * (1.2067492 + y_i0_2 * (0.2659732 + y_i0_2 * (0.0360768 + y_i0_2 * 0.0045813)))))

    ln_half = -0.6931471805599453  # -ln(2)
    log_term = ln_half + tl.log(tl.maximum(x_f32, 1e-10))
    ans_small = p_series - log_term * i0

    # Large x region: x > 2
    # Asymptotic series from mpmath (more accurate for larger x)
    inv_x = 1.0 / ax
    inv_x2 = inv_x * inv_x
    inv_x3 = inv_x2 * inv_x
    inv_x4 = inv_x3 * inv_x
    inv_x5 = inv_x4 * inv_x

    # Asymptotic series from mpmath - works better for x > 2
    # K0(x) ~ sqrt(pi/(2x)) * exp(-x) * (1 - 1/(4x) + 3/(64x^2) - 25/(1536x^3) + ...)
    P = 1.0 + inv_x * (-0.25) + inv_x2 * 0.0703125 + inv_x3 * (-0.05612890625) + inv_x4 * 0.0574470901517862 + inv_x5 * (-0.0629281194458016)

    pref = tl.sqrt(1.5707963267948966 / ax) * tl.exp(-ax)
    ans_large = pref * P

    # Select based on x value - use 2.0 as boundary (small x uses series, large x uses asymptotic)
    is_small = x_f32 <= 2.0
    ans = tl.where(is_small, ans_small, ans_large)

    # Handle x = 0 case: return infinity
    ans = tl.where(ax < 1e-10, float("inf"), ans)

    # Cast back to input dtype and store
    tl.store(out_ptr + offsets, ans.to(x.dtype), mask=mask)


def _launch_special_modified_bessel_k0(x: torch.Tensor, out: torch.Tensor):
    assert x.is_cuda and out.is_cuda, "Tensors must be CUDA tensors"
    assert x.numel() == out.numel(), "Input and output must have the same number of elements"
    assert x.dtype == out.dtype, "Input and output must have the same dtype"

    n_elements = x.numel()
    if n_elements == 0:
        return

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        special_modified_bessel_k0_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)


def special_modified_bessel_k0(self: torch.Tensor):
    logger.debug("ILUVATAR GEMS SPECIAL_MODIFIED_BESSEL_K0")
    x = self
    if not x.is_cuda:
        raise ValueError("special_modified_bessel_k0: input tensor must be on CUDA device")

    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    _launch_special_modified_bessel_k0(x_c, out)

    if x.layout == torch.strided and x.is_contiguous():
        return out
    else:
        return out.view_as(x)


def special_modified_bessel_k0_out(self: torch.Tensor, out: torch.Tensor):
    logger.debug("ILUVATAR GEMS SPECIAL_MODIFIED_BESSEL_K0_OUT")
    x = self
    if not (x.is_cuda and out.is_cuda):
        raise ValueError("special_modified_bessel_k0_out: input and output tensors must be on CUDA device")
    if not out.is_floating_point():
        raise TypeError("special_modified_bessel_k0_out: output tensor must be a floating point type")
    if x.numel() != out.numel():
        raise ValueError("special_modified_bessel_k0_out: input and output must have the same number of elements")

    x_c = x.contiguous()
    out_c = out.contiguous()
    _launch_special_modified_bessel_k0(x_c, out_c)

    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)
    return out