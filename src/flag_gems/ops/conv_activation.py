import logging

import torch

import flag_gems

logger = logging.getLogger(__name__)


def conv_activation(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
):
    """
    Conv2d followed by ReLU activation.

    This is a fused operation that performs convolution and then applies ReLU activation.
    """
    logger.debug("GEMS CONV_ACTIVATION")
    # Call conv2d
    conv_out = flag_gems.conv2d(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    # Apply relu
    return flag_gems.relu(conv_out)