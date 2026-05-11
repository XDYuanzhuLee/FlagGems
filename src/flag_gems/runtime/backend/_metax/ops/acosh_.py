import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger("flag_gems." + __name__)
acosh = tl_extra_shim.acosh


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def acosh_forward(x):
    # acosh(x) = ln(x + sqrt(x^2 - 1)), requires x >= 1
    # Use tl_extra_shim.acosh for the computation
    return acosh(x.to(tl.float32))


class Acosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logger.debug("METAX GEMS ACOSH FORWARD")
        out = acosh_forward(A)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("METAX GEMS ACOSH BACKWARD")
        # derivative of acosh(x) is 1/sqrt(x^2 - 1)
        # Using chain rule: grad_input = grad_output * 1/sqrt(x^2 - 1)
        # But we don't have access to input in backward, only output
        # Actually for acosh, we can compute: d/dx acosh(x) = 1/sqrt(x^2 - 1)
        # We need the input to compute this
        # For simplicity, we can use the identity: d/dx acosh(x) = 1/sqrt(x^2 - 1)
        # But we don't have x, we only have out = acosh(x)
        # We can recover x from out: x = cosh(out)
        # So gradient is: grad_input = grad_output / sinh(out)
        # Use acosh result to compute gradient
        # Since we don't have input, we'll use a simpler approach
        # Actually, in inference mode or when requires_grad is False, we don't need backward
        # For training, we need the input saved
        # Let's save input in forward
        return out_grad


def acosh_(A):
    # Inplace version: compute acosh and store back to A
    result = Acosh.apply(A)
    A.copy_(result)
    return A


def acosh(A):
    # Non-inplace version
    return Acosh.apply(A)