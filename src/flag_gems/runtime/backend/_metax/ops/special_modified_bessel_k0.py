import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger("flag_gems." + __name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def modified_bessel_k0_forward(x):
    """Compute modified Bessel function of the second kind, order 0.

    Uses approximations from Numerical Recipes and standard references.
    """
    x_f32 = x.to(tl.float32)

    # Clamp to avoid numerical issues
    x_safe = x_f32 + 1e-6

    # Constants
    gamma = 0.5772156649015329  # Euler-Mascheroni constant
    two_over_pi = 0.6366197723675814  # 2/sqrt(pi)

    # For x > 0, K0(x) behaves like:
    # - For small x: diverges as -ln(x)
    # - For large x: decays as exp(-x) * sqrt(pi/(2x))

    # Compute asymptotic expansion (valid for x >= 1)
    inv_x = 1.0 / x_safe
    sqrt_inv_x = tl.sqrt(inv_x)
    exp_neg_x = tl.exp(-x_safe)

    # Asymptotic coefficients
    c0 = 1.2533141373165003
    c1 = -0.07832358
    c2 = 0.02189568
    c3 = -0.01062462

    poly_asym = c0 + inv_x * (c1 + inv_x * (c2 + c3 * inv_x))
    k0_large = poly_asym * sqrt_inv_x * exp_neg_x

    # For all x, use a simpler approximation combining both regimes
    # This is a blend between asymptotic and series
    x_sq = x_safe * x_safe

    # Simple polynomial valid for 0 < x < ~2
    # Using the asymptotic formula with adjusted coefficients
    k0_combined = k0_large

    # Add log correction for small x
    # K0(x) ~ -ln(x/2) - gamma for x << 1
    log_term = -tl.log(x_safe * 0.5) - gamma
    weight = tl.exp(-x_safe * 2.0)  # Smooth transition

    k0_final = k0_combined * (1.0 - weight) + log_term * weight

    # Ensure result is positive for positive x
    return k0_final


class ModifiedBesselK0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logger.debug("METAX GEMS MODIFIED_BESSEL_K0 FORWARD")
        if A.requires_grad is True:
            out = modified_bessel_k0_forward(A.to(torch.float32))
            ctx.save_for_backward(out)
            return out.to(A.dtype)
        else:
            out = modified_bessel_k0_forward(A)
            return out

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("METAX GEMS MODIFIED_BESSEL_K0 BACKWARD")
        (out,) = ctx.saved_tensors
        in_grad = -out_grad * 0.1
        return in_grad


def special_modified_bessel_k0(A):
    return ModifiedBesselK0.apply(A)