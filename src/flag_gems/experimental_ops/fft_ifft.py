import logging

import torch

logger = logging.getLogger("flag_gems." + __name__)


def fft_ifft(input: torch.Tensor, n=None, dim=-1, norm=None):
    """
    Metax specialized inverse FFT operation.

    This implementation delegates to PyTorch's FFT which uses
    the underlying hardware FFT library (cuFFT on NVIDIA, etc.)

    Note: This function uses the raw PyTorch FFT API to avoid
    recursion when used within the flag_gems dispatch system.
    """
    logger.debug("METAX GEMS IFFT")
    # Ensure input is on GPU
    if not input.is_cuda:
        raise ValueError("fft_ifft requires CUDA input")

    # Handle complex types - FFT requires complex input
    if input.dtype not in [torch.complex64, torch.complex128]:
        raise TypeError(
            f"fft_ifft requires complex dtype input, got {input.dtype}"
        )

    # Call the raw PyTorch FFT function directly
    # This is the built-in function from torch._C._fft
    result = torch._C._fft.fft_ifft(input, n, dim, norm)

    return result