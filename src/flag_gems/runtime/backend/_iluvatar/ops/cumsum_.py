import logging

import torch
from torch._prims_common import is_boolean_dtype, is_integer_dtype

logger = logging.getLogger(__name__)


def cumsum_(inp, dim=1, *, dtype=None):
    """
    In-place cumulative sum operator for Iluvatar backend.
    Uses the generic cumsum implementation and copies result back to input.
    """
    logger.debug("ILUVATAR GEMS CUMSUM_")

    # Determine output dtype - for integers, keep input dtype (Iluvatar behavior)
    if dtype is None:
        dtype = inp.dtype
        if is_integer_dtype(dtype) or is_boolean_dtype(dtype):
            # Iluvatar's cumsum_ keeps input dtype
            pass

    # Call generic cumsum and copy result back
    from flag_gems.ops.cumsum import cumsum as generic_cumsum
    result = generic_cumsum(inp, dim=dim, dtype=dtype)
    inp.copy_(result)
    return inp