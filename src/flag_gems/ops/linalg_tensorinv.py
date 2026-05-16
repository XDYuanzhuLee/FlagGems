import logging

import torch

logger = logging.getLogger(__name__)


def tensorinv(A: torch.Tensor, ind: int = 2) -> torch.Tensor:
    logger.debug("GEMS LINALG_TENSORINV")
    return torch.linalg.tensorinv(A, ind=ind)