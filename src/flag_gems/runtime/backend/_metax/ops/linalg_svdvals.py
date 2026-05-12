import logging

import torch

logger = logging.getLogger("flag_gems." + __name__)


def linalg_svdvals(A: torch.Tensor, driver: str = None) -> torch.Tensor:
    """Computes the singular values of a matrix.

    This is a Metax backend specialization that delegates to torch.linalg.svdvals.

    Args:
        A: Input tensor of shape (*, m, n) where * is zero or more batch dimensions.
        driver: Optional cuSOLVER method (not used on Metax).

    Returns:
        Singular values in descending order, shape (*, min(m, n)).
    """
    logger.debug("METAX GEMS LINALG_SVDVALS")
    # Ensure input is contiguous
    if not A.is_contiguous():
        A = A.contiguous()

    # Call torch.linalg.svdvals
    # Note: On CUDA devices, this will synchronize
    result = torch.linalg.svdvals(A, driver=driver)
    return result