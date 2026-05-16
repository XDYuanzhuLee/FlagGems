import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)
exp2 = tl_extra_shim.exp2


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def special_expit_func(x):
    log2e: tl.constexpr = 1.4426950408889634
    return 1 / (1 + exp2(-x.to(tl.float32) * log2e))


def special_expit(A):
    logger.debug("ILUVATAR GEMS special_expit")
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A)
    return special_expit_func(A)


def special_expit_(A):
    logger.debug("ILUVATAR GEMS special_expit_")
    if isinstance(A, torch.Tensor):
        return special_expit_func(A, out0=A)
    else:
        A = torch.tensor(A)
        return special_expit_func(A, out0=A)