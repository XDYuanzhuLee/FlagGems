import logging

import torch

logger = logging.getLogger(__name__)


def conv_transpose1d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    groups: int = 1,
    dilation: int = 1,
) -> torch.Tensor:
    """
    Applies a 1D transposed convolution operator over an input image
    composed of several input planes.

    This is a wrapper around torch.nn.functional.conv_transpose1d.
    """
    logger.debug("GEMS CONV_TRANSPOSE1D")
    return torch.nn.functional.conv_transpose1d(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )