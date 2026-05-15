import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)

div_rn = tl_extra_shim.div_rn


@pointwise_dynamic(
    is_tensor=[True, True, True, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def addcdiv_kernel(inp, t1, t2, value):
    return inp + value * (t1 / t2)


@pointwise_dynamic(
    is_tensor=[True, True, True, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def addcdiv_kernel_v2(inp, t1, t2, value):
    result = inp + value * div_rn(t1, t2)
    return result


def addcdiv_(inp, tensor1, tensor2, value=1.0):
    logger.debug("ILUVATAR GEMS ADDCDIV_")
    if isinstance(tensor1, torch.Tensor) and isinstance(tensor2, torch.Tensor):
        return addcdiv_kernel(inp, tensor1, tensor2, value, out0=inp)
    else:
        return addcdiv_kernel(inp, tensor1, tensor2, value, out0=inp)