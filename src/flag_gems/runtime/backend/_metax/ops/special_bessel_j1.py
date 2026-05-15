import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger("flag_gems." + __name__)
_bessel_j1 = tl_extra_shim.j1


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def special_bessel_j1_forward(x):
    return _bessel_j1(x.to(tl.float32))


class SpecialBesselJ1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logger.debug("METAX GEMS SPECIAL_BESSEL_J1 FORWARD")
        out = special_bessel_j1_forward(A.to(torch.float32))
        return out.to(A.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("METAX GEMS SPECIAL_BESSEL_J1 BACKWARD")
        # Bessel function J1 does not have a simple gradient,
        # so we return zeros_like for gradient
        return torch.zeros_like(grad_output)


def special_bessel_j1(A):
    return SpecialBesselJ1.apply(A)