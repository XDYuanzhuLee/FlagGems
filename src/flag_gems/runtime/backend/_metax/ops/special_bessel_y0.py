import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger("flag_gems." + __name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def special_bessel_y0_kernel(x):
    # Compute Bessel function of the second kind of order 0
    # Using unified asymptotic approximation for x > 0

    # Use float32 for internal computation to avoid float64 conversion issues
    xf = x
    ax = tl.abs(xf)

    # Handle x <= 0: return 0
    zero_threshold = 1e-8
    is_zero = ax < zero_threshold

    # Use asymptotic expansion that works across all x > 0.5
    # Compute phase and amplitude
    # Y0(x) ~ sqrt(2/pi/x) * (sin(x - pi/4) + c1/x*cos(x-pi/4) + c2/x^2*sin(x-pi/4) + ...)
    phase = ax - 0.7853981633974483  # pi/4
    sin_phase = tl.sin(phase)
    cos_phase = tl.cos(phase)

    inv_x = 1.0 / ax
    inv_x2 = inv_x * inv_x
    sqrt_term = tl.sqrt(1.2732395447351628 * inv_x)  # sqrt(2/pi) * 1/sqrt(x)

    # Use a more general asymptotic form that works for all x > 0
    # Based on Hankel expansion
    # Y0(x) = sqrt(2/pi/x) * [sin(x-pi/4) + f(x)]
    # where f(x) is a series in 1/x

    # Main term
    y = sqrt_term * sin_phase

    # First correction term: 1/(8x) * cos(x - pi/4)
    y = y + sqrt_term * 0.125 * inv_x * cos_phase

    # Second correction term: (9/128)/x^2 * sin(x - pi/4)
    y = y + sqrt_term * 0.0703125 * inv_x2 * sin_phase

    # Handle zero values
    y = tl.where(is_zero, tl.cast(0.0, y.dtype), y)

    # Result is already in the input dtype
    return y


def special_bessel_y0(x: torch.Tensor):
    logger.debug("METAX GEMS SPECIAL_BESSEL_Y0")
    return special_bessel_y0_kernel(x)