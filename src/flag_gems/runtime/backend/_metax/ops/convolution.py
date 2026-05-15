import logging

import torch

logger = logging.getLogger("flag_gems." + __name__)


def convolution(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    transposed=False,
    output_padding=0,
    groups=1,
):
    """
    Applies a convolution over an input tensor.

    This is a specialized implementation for Metax GPUs that delegates to
    the underlying torch convolution implementation.
    """
    logger.debug("METAX GEMS CONVOLUTION")

    # Validate inputs
    assert weight.ndim in [3, 4, 5], (
        f"Weights must be 3D, 4D or 5D, received shape {weight.shape}"
    )
    assert (
        bias is None or bias.ndim == 1
    ), f"Bias must be 1D, received shape {bias.shape}"

    # Handle stride - convert to list/tuple
    if isinstance(stride, int):
        stride = (stride,) * (weight.ndim - 2)
    elif hasattr(stride, '__len__') and len(stride) == 1:
        stride = tuple(stride * (weight.ndim - 2))
    else:
        stride = tuple(stride)

    # Handle padding
    if isinstance(padding, int):
        padding = (padding,) * (weight.ndim - 2)
    elif hasattr(padding, '__len__') and len(padding) == 1:
        padding = tuple(padding * (weight.ndim - 2))
    else:
        padding = tuple(padding)

    # Handle dilation
    if isinstance(dilation, int):
        dilation = (dilation,) * (weight.ndim - 2)
    elif hasattr(dilation, '__len__') and len(dilation) == 1:
        dilation = tuple(dilation * (weight.ndim - 2))
    else:
        dilation = tuple(dilation)

    # Handle output_padding (only for transposed conv)
    if isinstance(output_padding, int):
        output_padding = (output_padding,) * (weight.ndim - 2)
    elif hasattr(output_padding, '__len__') and len(output_padding) == 1:
        output_padding = tuple(output_padding * (weight.ndim - 2))
    elif output_padding:
        output_padding = tuple(output_padding)
    else:
        output_padding = (0,) * (weight.ndim - 2)

    # Delegate to torch convolution
    return torch.ops.aten.convolution(
        input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
    )