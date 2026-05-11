import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.sum import sum as generic_sum, sum_dim as generic_sum_dim
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)


@libentry()
@triton.jit
def sum_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M

    inp_val = tl.load(inp_ptrs, mask=mask, other=0).to(cdtype)
    sum_val = tl.sum(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, sum_val)


@libentry()
@triton.jit
def sum_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    if tl.constexpr(mid.dtype.element_ty == tl.float16) or tl.constexpr(
        mid.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = mid.dtype.element_ty

    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=0).to(cdtype)
    sum_val = tl.sum(mid_val)
    tl.store(out, sum_val)


def sum(inp, *, dtype=None):
    """Metax specialized sum - reduces all elements to a scalar."""
    logger.debug("METAX GEMS SUM")
    inp = inp.contiguous()
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            inp = inp.to(torch.int64)
            dtype = torch.int64
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        sum_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
        sum_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out


def sum_dim(inp, dim=None, keepdim=False, *, dtype=None):
    """Metax specialized sum_dim - delegates to generic for complex cases."""
    logger.debug("METAX GEMS SUM DIM")
    # For complex cases (multiple dimensions, specific edge cases),
    # delegate to generic implementation
    return generic_sum_dim(inp, dim, keepdim, dtype=dtype)