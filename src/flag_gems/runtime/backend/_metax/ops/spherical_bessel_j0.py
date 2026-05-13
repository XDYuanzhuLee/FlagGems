import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger("flag_gems." + __name__)
sin = tl_extra_shim.sin
cos = tl_extra_shim.cos


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def spherical_bessel_j0_forward(x):
    # j0(x) = sin(x) / x for x != 0
    # j0(0) = 1 (limit as x -> 0)
    x_fp32 = x.to(tl.float32)
    # Handle x near zero: use approximation sin(x) ≈ x for small x
    # This gives j0(x) ≈ 1 for small x
    is_small = tl.abs(x_fp32) < tl.constexpr(1e-6)
    small_result = tl.constexpr(1.0)
    # For larger x, compute sin(x) / x
    large_result = sin(x_fp32) / x_fp32
    return tl.where(is_small, small_result, large_result)


class SphericalBesselJ0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logger.debug("METAX GEMS SPHERICAL_BESSEL_J0 FORWARD")
        if A.requires_grad is True:
            out = spherical_bessel_j0_forward(A.to(torch.float32))
            ctx.save_for_backward(out)
            return out.to(A.dtype)
        else:
            out = spherical_bessel_j0_forward(A)
            return out

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("METAX GEMS SPHERICAL_BESSEL_J0 BACKWARD")
        (out,) = ctx.saved_tensors
        # Derivative of j0(x) = sin(x)/x:
        # dj0/dx = (x*cos(x) - sin(x)) / x^2
        # For small x: dj0/dx ≈ -x/3
        raise NotImplementedError("spherical_bessel_j0 backward is not implemented yet")


def spherical_bessel_j0(A):
    return SphericalBesselJ0.apply(A)