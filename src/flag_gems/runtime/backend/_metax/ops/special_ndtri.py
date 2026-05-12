import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger("flag_gems." + __name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def special_ndtri_forward(x):
    # ndtri(p) = sqrt(2) * erfinv(2p - 1)
    # Transform p from [0, 1] to [-1, 1]
    p = x.to(tl.float32)
    two_p_minus_1 = p * 2.0 - 1.0

    # Compute erfinv using the same approximation as erfinv.py
    one = 1.0
    absx = tl.abs(two_p_minus_1)
    w = -tl.log((one - two_p_minus_1) * (one + two_p_minus_1))

    use_low = w < 5.0

    wl = w - 2.5
    pl = 2.81022636e-08
    pl = 3.43273939e-07 + pl * wl
    pl = -3.5233877e-06 + pl * wl
    pl = -4.39150654e-06 + pl * wl
    pl = 2.1858087e-04 + pl * wl
    pl = -1.25372503e-03 + pl * wl
    pl = -4.17768164e-03 + pl * wl
    pl = 2.46640727e-01 + pl * wl
    pl = 1.50140941e00 + pl * wl

    wh = tl.sqrt(w) - 3.0
    ph = -2.00214257e-04
    ph = 1.00950558e-04 + ph * wh
    ph = 1.34934322e-03 + ph * wh
    ph = -3.67342844e-03 + ph * wh
    ph = 5.73950773e-03 + ph * wh
    ph = -7.62246130e-03 + ph * wh
    ph = 9.43887047e-03 + ph * wh
    ph = 1.00167406e00 + ph * wh
    ph = 2.83297682e00 + ph * wh

    erfinv_val = tl.where(use_low, pl, ph)
    erfinv_val = erfinv_val * two_p_minus_1

    # Multiply by sqrt(2)
    sqrt2 = 1.4142135623730951
    result = erfinv_val * sqrt2

    # Handle edge cases
    nan_vec = tl.full([1], float("nan"), dtype=tl.float32)
    inf_vec = tl.full([1], float("inf"), dtype=tl.float32)
    neg_inf_vec = tl.full([1], float("-inf"), dtype=tl.float32)

    # p == 0 -> -inf, p == 1 -> inf, p < 0 or p > 1 -> nan
    mask_nan = p != p  # isnan
    mask_oob = (p < 0.0) | (p > 1.0)
    mask_zero = p == 0.0
    mask_one = p == 1.0

    result = tl.where(mask_nan, nan_vec, result)
    result = tl.where(mask_oob, nan_vec, result)
    result = tl.where(mask_zero, neg_inf_vec, result)
    result = tl.where(mask_one, inf_vec, result)

    return result


def special_ndtri(x: torch.Tensor):
    logger.debug("METAX GEMS SPECIAL_NDTRI")
    return special_ndtri_forward(x)