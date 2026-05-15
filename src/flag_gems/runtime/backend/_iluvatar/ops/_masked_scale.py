import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger("flag_gems." + __name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _masked_scale_kernel(input, mask, scale):
    # mask is uint8: 1 means keep (multiply by scale), 0 means zero out
    return tl.where(mask != 0, input * scale, 0.0)


def _masked_scale(input, mask, scale):
    logger.debug("ILUVATAR GEMS _masked_scale")
    return _masked_scale_kernel(input, mask, scale)