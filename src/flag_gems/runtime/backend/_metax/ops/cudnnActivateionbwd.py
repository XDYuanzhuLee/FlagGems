import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger("flag_gems." + __name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def cudnn_activation_backward_kernel(x, dy):
    """ReLU backward: gradient is passed through for x > 0"""
    return tl.where(x > 0, dy, 0)


def cudnnActivateionbwd(x, grad_output):
    """Cudnn activation backward (ReLU backward) operator.

    Computes the gradient of the input with respect to the ReLU activation.

    Args:
        x: The input tensor (output of the forward pass)
        grad_output: The gradient of the loss with respect to the output

    Returns:
        The gradient of the loss with respect to the input
    """
    logger.debug("METAX GEMS CUDNNACTIVATIONBWD")
    return cudnn_activation_backward_kernel(x, grad_output)