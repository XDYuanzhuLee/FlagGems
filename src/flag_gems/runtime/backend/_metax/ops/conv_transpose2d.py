import logging

import torch

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger("flag_gems." + __name__)


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

    Metax specialized version that wraps torch.nn.functional.conv_transpose1d.
    """
    logger.debug("METAX GEMS CONV_TRANSPOSE1D")
    with torch_device_fn.device(input.device):
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


def conv_transpose2d(
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
    Applies a 2D transposed convolution operator over an input image
    composed of several input planes.

    Metax specialized version that wraps torch.nn.functional.conv_transpose2d.
    """
    logger.debug("METAX GEMS CONV_TRANSPOSE2D")
    with torch_device_fn.device(input.device):
        return torch.nn.functional.conv_transpose2d(
            input,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        )