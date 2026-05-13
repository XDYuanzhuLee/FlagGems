import logging

import torch

logger = logging.getLogger("flag_gems." + __name__)


def _pin_memory(A: torch.Tensor, device=None):
    """Pin memory operator for Metax GPU backend.

    For GPU tensors, this is effectively a no-op since GPU memory
    is already pinned/fixed. We simply return the input tensor.
    """
    logger.debug("METAX GEMS _PIN_MEMORY")
    # For GPU tensors, return as-is since GPU memory is already pinned
    return A