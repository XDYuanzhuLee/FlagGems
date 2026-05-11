import logging

import torch

logger = logging.getLogger("flag_gems." + __name__)


def shape(inp) -> torch.Tensor:
    """
    Returns the shape of a tensor as a 1D int64 tensor.

    This is similar to ONNX Shape operator and PyTorch's _shape_as_tensor.

    Args:
        inp: Input tensor.

    Returns:
        A 1D int64 tensor containing the shape of the input tensor.
    """
    logger.debug("METAX GEMS SHAPE")
    return torch.tensor(inp.shape, dtype=torch.int64, device=inp.device)