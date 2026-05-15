import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def log10_func(x):
    # log10(x) = log(x) / log(10) = log(x) / ln(10)
    LN10 = 2.3025850929940459
    return tl.log(x.to(tl.float32)) / LN10


def log10(A):
    logger.debug("ILUVATAR GEMS LOG10")
    return log10_func(A)