import logging

import torch

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger("flag_gems." + __name__)


def tensorinv(A: torch.Tensor, ind: int = 2) -> torch.Tensor:
    logger.debug("METAX GEMS LINALG_TENSORINV")
    with torch_device_fn.device(A.device):
        return torch.linalg.tensorinv(A, ind=ind)