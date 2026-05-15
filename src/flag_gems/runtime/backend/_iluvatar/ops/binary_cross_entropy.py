import logging
import math
from enum import Enum

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)

logger = logging.getLogger("flag_gems." + __name__)


@libentry()
@triton.jit
def kernel_1(inp, target, mid, M, BLOCK_SIZE: tl.constexpr, reduction: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    target_ptrs = target + offset
    mask = offset < M

    inp_val = tl.load(inp_ptrs, mask=mask, other=0.0).to(tl.float32)
    target_val = tl.load(target_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Binary cross entropy: -(target * log(inp) + (1 - target) * log(1 - inp))
    # Clip inputs to avoid log(0)
    eps = 1e-7
    inp_clipped = tl.where(inp_val < eps, eps, tl.where(inp_val > 1 - eps, 1 - eps, inp_val))
    target_clipped = tl.where(target_val < eps, eps, tl.where(target_val > 1 - eps, 1 - eps, target_val))

    loss = -(target_clipped * tl.log(inp_clipped) + (1 - target_clipped) * tl.log(1 - inp_clipped))

    # Reduction.MEAN.value: 1 Reduction.SUM.value: 2
    if reduction == 1:
        sum_val = tl.sum(loss) / M
    else:
        sum_val = tl.sum(loss)
    mid_ptr = mid + pid
    tl.store(mid_ptr, sum_val)


@libentry()
@triton.jit
def kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr, output_dtype: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=0.0).to(tl.float32)
    sum_val = tl.sum(mid_val)
    # Convert to output dtype to avoid overflow
    if output_dtype == tl.float16:
        sum_val = sum_val.to(tl.float16)
    elif output_dtype == tl.bfloat16:
        sum_val = sum_val.to(tl.bfloat16)
    tl.store(out, sum_val)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def func(x, y):
    # Binary cross entropy: -(target * log(inp) + (1 - target) * log(1 - inp))
    # Clip inputs to avoid log(0) using tl.where
    eps = 1e-7
    x_clipped = tl.where(x < eps, eps, tl.where(x > 1 - eps, 1 - eps, x))
    y_clipped = tl.where(y < eps, eps, tl.where(y > 1 - eps, 1 - eps, y))
    return -(y_clipped * tl.log(x_clipped) + (1 - y_clipped) * tl.log(1 - x_clipped))


class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


def binary_cross_entropy(inp, target, weight=None, reduction=Reduction.MEAN.value):
    logger.debug("ILUVATAR GEMS BINARY_CROSS_ENTROPY")
    # TODO: implement weight support
    if reduction == Reduction.NONE.value:
        return func(inp, target)

    inp = inp.contiguous()
    target = target.contiguous()
    M = inp.numel()
    dtype = inp.dtype

    # For mean/sum reduction, use float32 output to avoid precision issues
    output_dtype = tl.float32

    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.float32, device=inp.device)
    out = torch.empty([], dtype=torch.float32, device=inp.device)

    with torch_device_fn.device(inp.device):
        kernel_1[(mid_size, 1, 1)](inp, target, mid, M, block_size, reduction)
        kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid, output_dtype)

    # Convert back to original dtype
    return out.to(dtype)