import logging

import torch

logger = logging.getLogger(__name__)


def cudnnconvfilter(
    input: torch.Tensor,
    weight: torch.Tensor,
    padding=0,
    stride=1,
    dilation=1,
    groups=1,
):
    """
    Applies a 2D convolution filter to the input tensor.

    This is a wrapper around torch.nn.functional.conv2d with cuDNN-specific defaults.

    Args:
        input: Input tensor of shape (N, C, H, W)
        weight: Weight tensor of shape (out_channels, in_channels, kH, kW)
        padding: Padding applied to input
        stride: Stride of the convolution
        dilation: Dilation of the convolution
        groups: Number of groups for grouped convolution

    Returns:
        Output tensor
    """
    logger.debug("GEMS CUDNNCONVFILTER")
    return torch.nn.functional.conv2d(
        input,
        weight,
        padding=padding,
        stride=stride,
        dilation=dilation,
        groups=groups,
    )