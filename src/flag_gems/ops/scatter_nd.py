import logging

import torch

logger = logging.getLogger(__name__)


def scatter_nd(inp, indices, values, accumulate=False):
    """Scatter values into a tensor at the given indices.

    This is a wrapper that falls back to torch.index_put.
    The actual implementation should be provided by a specialized backend.

    Args:
        inp: The input tensor to scatter into.
        indices: An integer tensor of shape (..., rank) where rank is the
                 number of dimensions in the output tensor.
        values: The values to scatter.
        accumulate: If True, accumulate instead of overwrite.

    Returns:
        The output tensor with scattered values.
    """
    logger.debug("GEMS SCATTER_ND")

    # Convert scatter_nd style indices to index_put style (list of tensors)
    indices_list = indices.unbind(-1)

    if accumulate:
        return torch.index_put(inp, indices_list, values, accumulate=True)
    else:
        return torch.index_put(inp, indices_list, values, accumulate=False)