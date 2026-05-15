import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger("flag_gems." + __name__)


@triton.jit
def reduce_all(a, b):
    return a and b


@triton.jit
def is_all_true_kernel_1(
    inp,
    mid,
    n_elements,
    mid_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < n_elements
    inp_val = tl.load(inp_ptrs, mask=mask, other=1.0)
    all_val = tl.reduce(inp_val != 0, axis=0, combine_fn=reduce_all)
    mid_ptr = mid + pid
    tl.store(mid_ptr, all_val)


@triton.jit
def is_all_true_kernel_2(mid, out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptrs, mask=mask, other=1).to(tl.int1)
    all_val = tl.reduce(mid_val, axis=0, combine_fn=reduce_all)
    tl.store(out, all_val)


def _is_all_true(inp):
    logger.debug("ILUVATAR GEMS _IS_ALL_TRUE")
    n_elements = inp.numel()
    # Empty tensor returns True (all of empty set is True)
    if n_elements == 0:
        return torch.tensor(True, dtype=torch.bool, device=inp.device)
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    mid_size = triton.cdiv(n_elements, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=torch.bool, device=inp.device)
    out = torch.empty([], dtype=torch.bool, device=inp.device)

    with torch_device_fn.device(inp.device):
        is_all_true_kernel_1[(mid_size, 1)](inp, mid, n_elements, mid_size, block_size)
        is_all_true_kernel_2[(1, 1)](mid, out, mid_size, block_mid)

    return out