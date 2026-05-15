import logging

import torch

logger = logging.getLogger(__name__)


def _unsafe_masked_index_put_accumulate(input, mask, indices, values):
    """
    Generic implementation of _unsafe_masked_index_put_accumulate.
    Falls back to torch native implementation.
    """
    logger.debug("GEMS _UNSAFE_MASKED_INDEX_PUT_ACCUMULATE")
    return torch._unsafe_masked_index_put_accumulate(input, mask, indices, values)