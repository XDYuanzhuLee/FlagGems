import logging
import math
from collections import namedtuple

import torch
import triton

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger("flag_gems." + __name__)


def median(inp, dim=None, keepdim=False):
    """Compute median of tensor.

    If dim is None, returns the median of all elements.
    Otherwise, computes median along the specified dimension.
    Returns (values, indices) namedtuple.
    """
    logger.debug("METAX GEMS MEDIAN")

    # Use torch's native median implementation for correctness
    # This will use the GPU's native median operation
    result = torch.median(inp, dim=dim, keepdim=keepdim)

    if dim is None:
        # Returns a scalar tensor
        return result
    else:
        # Returns namedtuple (values, indices)
        Median_out = namedtuple("median", ["values", "indices"])
        return Median_out(values=result.values, indices=result.indices)