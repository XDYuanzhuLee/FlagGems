import logging

import torch

logger = logging.getLogger("flag_gems." + __name__)


def linalg_slogdet(A):
    """Metax specialized implementation of linalg_slogdet.

    This function computes the sign and natural logarithm of the absolute value
    of the determinant of a square matrix using torch.linalg.slogdet.

    Args:
        A: Input tensor of shape (*, n, n) where * is zero or more batch dimensions.

    Returns:
        A named tuple (sign, logabsdet) where:
            - sign: tensor with the same dtype as A
            - logabsdet: real-valued tensor
    """
    logger.debug("METAX GEMS linalg_slogdet")
    result = torch.linalg.slogdet(A)
    return result


def _linalg_slogdet_aten(A):
    """ATen compatibility wrapper for linalg_slogdet.

    The aten::_linalg_slogdet returns (sign, logabsdet, LU, pivots) but
    torch.linalg.slogdet only returns (sign, logabsdet). This wrapper
    maintains compatibility with the aten schema.

    Args:
        A: Input tensor of shape (*, n, n)

    Returns:
        Tuple of (sign, logabsdet, LU, pivots) where LU and pivots are empty tensors
    """
    logger.debug("METAX GEMS _linalg_slogdet_aten")
    sign, logabsdet = torch.linalg.slogdet(A)
    # Create empty tensors for LU and pivots to match aten schema
    n = A.shape[-1]
    LU = torch.empty_like(A)
    pivots = torch.empty((*A.shape[:-2], n), dtype=torch.int32, device=A.device)
    return sign, logabsdet, LU, pivots