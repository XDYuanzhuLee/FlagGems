import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils import tl_extra_shim

sqrt = tl_extra_shim.sqrt
logger = logging.getLogger("flag_gems." + __name__)


@libentry()
@triton.jit
def reduce_l2_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=0.0).to(tl.float32)
    sum_val = tl.sum(inp_val * inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, sum_val)


@libentry()
@triton.jit
def reduce_l2_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=0.0).to(tl.float32)
    out_val = sqrt(tl.sum(mid_val))
    tl.store(out, out_val)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 1024}, num_stages=1, num_warps=8),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 512}, num_stages=1, num_warps=8),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 256}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_stages=1, num_warps=4),
    ],
    key=["M", "N"],
)
@triton.jit
def reduce_l2_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of inp it should compute.
    pid = tle.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask
        a = tl.load(inp + cols, mask, other=0.0).to(tl.float32)
        _sum += a * a
    sum_val = tl.sum(_sum, axis=1)
    out_val = sqrt(sum_val)[:, None]
    tl.store(out, out_val, row_mask)


def reduce_l2(inp, dim=None, keepdim=False):
    logger.debug("METAX GEMS REDUCE L2")
    # Convert dim to a list if it's an int
    dim_list = dim if dim is None or isinstance(dim, (list, tuple)) else [dim]
    if dim is None or len(dim_list) == 0:
        M = inp.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        dtype = inp.dtype
        mid = torch.empty((mid_size,), dtype=torch.float32, device=inp.device)
        if not keepdim:
            out = torch.empty([], dtype=dtype, device=inp.device)
        else:
            shape = [1] * inp.dim()
            out = torch.empty(shape, dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            reduce_l2_kernel_1[(mid_size, 1)](
                inp,
                mid,
                M,
                block_size,
            )
            reduce_l2_kernel_2[(1, 1)](
                mid, out, mid_size, block_mid
            )
        return out
    else:
        if isinstance(dim, int):
            dim = [dim]
        assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"
        dtype = inp.dtype

        shape = list(inp.shape)
        dim = [d % inp.ndim for d in dim]
        inp = dim_compress(inp, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N

        out = torch.empty(shape, dtype=dtype, device=inp.device)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(inp.device):
            reduce_l2_kernel[grid](inp, out, M, N)
        if not keepdim:
            out = out.squeeze(dim=dim)
        return out