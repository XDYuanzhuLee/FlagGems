import logging

import torch

logger = logging.getLogger("flag_gems." + __name__)


def linalg_det(input):
    """
    Compute the determinant of a batch of square matrices.
    This is a wrapper around torch._linalg_det that uses the METAX GEMS logger.

    Args:
        input: A square matrix or batch of square matrices (..., n, n)

    Returns:
        The determinant(s) as a tensor with shape (...)
    """
    logger.debug("METAX GEMS LINALG_DET")

    # Ensure input is contiguous
    input = input.contiguous()

    # Check input shape
    if input.ndim < 2:
        raise ValueError(f"Expected at least 2D input, got {input.ndim}D")

    n = input.shape[-1]
    if n != input.shape[-2]:
        raise ValueError(
            f"Expected a square matrix (or batch of square matrices), "
            f"got shape {input.shape}"
        )

    # Call torch._linalg_det and extract the result
    result = torch._linalg_det(input)
    return result.result