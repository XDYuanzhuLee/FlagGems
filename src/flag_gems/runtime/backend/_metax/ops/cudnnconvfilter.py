import logging

import torch

logger = logging.getLogger("flag_gems." + __name__)


def cudnnconvfilter(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    padding=0,
    stride=1,
    dilation=1,
    groups=1,
    benchmark=False,
    deterministic=False,
    allow_tf32=True,
):
    """
    Applies a 2D convolution filter to the input tensor.

    This is a metax-specialized implementation.

    Args:
        input: Input tensor of shape (N, C, H, W)
        weight: Weight tensor of shape (out_channels, in_channels, kH, kW)
        bias: Optional bias tensor of shape (out_channels,)
        padding: Padding applied to input
        stride: Stride of the convolution
        dilation: Dilation of the convolution
        groups: Number of groups for grouped convolution
        benchmark: Whether to use cuDNN benchmarking
        deterministic: Whether to use deterministic cuDNN
        allow_tf32: Whether to allow TF32 computation

    Returns:
        Output tensor
    """
    logger.debug("METAX GEMS CUDNNCONVFILTER")
    return torch.nn.functional.conv2d(
        input,
        weight,
        bias=bias,
        padding=padding,
        stride=stride,
        dilation=dilation,
        groups=groups,
    )