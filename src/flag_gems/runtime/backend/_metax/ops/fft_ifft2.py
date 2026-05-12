import logging

import torch

logger = logging.getLogger("flag_gems." + __name__)


def fft_ifft2(input, s=None, dim=(-2, -1), norm=None, *, out=None):
    """Metax specialized implementation of torch.fft.ifft2.

    This implementation delegates to PyTorch's FFT implementation with Metax-specific
    logging for debugging and profiling purposes.

    Args:
        input (Tensor): the input tensor
        s (Tuple[int], optional): Signal size in the transformed dimensions
        dim (Tuple[int], optional): Dimensions to be transformed. Default: last two dimensions
        norm (str, optional): Normalization mode. Default: "backward"
        out (Tensor, optional): the output tensor

    Returns:
        Tensor: The 2D inverse Fourier Transform
    """
    logger.debug("METAX GEMS FFT_IFFT2")
    # Delegate to PyTorch's ifft2 implementation
    result = torch.fft.ifft2(input, s=s, dim=dim, norm=norm, out=out)
    return result