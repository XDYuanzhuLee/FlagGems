import logging
from typing import Optional, Tuple, Union

import torch

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger("flag_gems." + __name__)


def fractional_max_pool2d(
    input: torch.Tensor,
    kernel_size: Tuple[int, int],
    output_size: Optional[Tuple[int, int]] = None,
    output_ratio: Optional[Tuple[float, float]] = None,
    return_indices: bool = False,
    _random_samples: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Fractional max pooling 2d for Metax backend.

    Applies 2D fractional max pooling over an input signal composed of several input planes.

    Args:
        input: Input tensor of shape (N, C, H, W)
        kernel_size: Size of the window to take a max over. Can be a single number or (kH, kW)
        output_size: Target output size (oH, oW). If None, will be computed from output_ratio
        output_ratio: If given, output size is computed as ratio of input size
        return_indices: If True, will return the indices along with outputs
        _random_samples: Pre-computed random samples for the pooling

    Returns:
        If return_indices is True, returns (output, indices), otherwise just output
    """
    logger.debug("METAX GEMS FRACTIONAL_MAX_POOL2D")

    with torch_device_fn.device(input.device):
        # Use torch.nn.functional.fractional_max_pool2d
        output = torch.nn.functional.fractional_max_pool2d(
            input,
            kernel_size=kernel_size,
            output_size=output_size,
            output_ratio=output_ratio,
            return_indices=return_indices,
            _random_samples=_random_samples,
        )

    return output