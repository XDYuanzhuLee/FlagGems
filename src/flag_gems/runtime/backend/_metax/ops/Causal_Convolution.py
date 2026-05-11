import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger("flag_gems." + __name__)


def causal_convolution(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> torch.Tensor:
    """Causal 1D convolution (Metax specialized version).

    A causal convolution ensures that output at position t can only depend on input at positions <= t.
    This is achieved by applying appropriate left padding.

    Args:
        input: Input tensor of shape (batch, in_channels, length)
        weight: Weight tensor of shape (out_channels, in_channels // groups, kernel_size)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Stride of the convolution
        padding: Additional padding on the left side
        dilation: Dilation of the convolution
        groups: Number of groups for grouped convolution

    Returns:
        Output tensor of shape (batch, out_channels, length)
    """
    logger.debug("METAX GEMS CAUSAL_CONVOLUTION")

    assert input.ndim == 3, f"Expected 3D input (batch, channels, length), got {input.ndim}D"
    assert weight.ndim == 3, f"Expected 3D weight (out_channels, in_channels, kernel_size), got {weight.ndim}D"

    batch, in_channels, length = input.shape
    out_channels, in_channels_per_group, kernel_size = weight.shape

    assert in_channels % groups == 0, "in_channels must be divisible by groups"
    assert out_channels % groups == 0, "out_channels must be divisible by groups"
    assert in_channels_per_group == in_channels // groups

    # Calculate causal padding
    # For causal convolution, we need left padding of (kernel_size - 1) * dilation
    # This ensures output[t] only depends on input[<=t]
    causal_padding = (kernel_size - 1) * dilation

    # Apply manual left padding (asymmetric) instead of using PyTorch's symmetric padding
    # F.pad with (left, right) - pad left side with causal_padding zeros, right side with user padding
    total_padding = causal_padding + padding
    padded_input = F.pad(input, (total_padding, 0), value=0.0)

    # Apply convolution with no additional padding
    output = F.conv1d(
        padded_input,
        weight,
        bias=bias,
        stride=stride,
        padding=0,  # No padding - we handled it manually
        dilation=dilation,
        groups=groups,
    )

    return output