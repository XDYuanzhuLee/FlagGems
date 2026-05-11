import logging

import torch

logger = logging.getLogger("flag_gems." + __name__)


def scatter_nd(inp, indices, values, accumulate=False):
    """Scatter values into a tensor at the given indices.

    This is the Metax specialized implementation that falls back to
    the reference implementation for now.

    Args:
        inp: The input tensor to scatter into.
        indices: An integer tensor of shape (..., rank) where rank is the
                 number of dimensions in the input tensor.
        values: The values to scatter.
        accumulate: If True, accumulate instead of overwrite.

    Returns:
        The output tensor with scattered values.
    """
    logger.debug("METAX GEMS SCATTER_ND")

    # Convert scatter_nd style indices to index_put style (list of tensors)
    indices_list = indices.unbind(-1)

    if accumulate:
        return torch.index_put(inp, indices_list, values, accumulate=True)
    else:
        return torch.index_put(inp, indices_list, values, accumulate=False)