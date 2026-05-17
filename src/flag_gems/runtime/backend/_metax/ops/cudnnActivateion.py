import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger("flag_gems." + __name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def cudnn_activation_forward_kernel(x):
    """ReLU forward: returns max(0, x)"""
    return tl.where(x > 0, x, 0)


def cudnnActivateion(x):
    """Cudnn activation forward (ReLU forward) operator.

    Applies the ReLU activation function: max(0, x).

    Args:
        x: The input tensor

    Returns:
        The output tensor with ReLU applied
    """
    logger.debug("METAX GEMS CUDNNACTIVATION")
    return cudnn_activation_forward_kernel(x)