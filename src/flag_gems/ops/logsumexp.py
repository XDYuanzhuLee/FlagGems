import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def logsumexp_kernel_1(
    inp,
    mid,
    max_ptr,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    max_val = tl.max(inp_val)
    tl.store(max_ptr + pid, max_val)

    # Compute exp(x - max)
    exp_val = tl.exp(inp_val - max_val)
    sum_val = tl.sum(exp_val, axis=0)
    log_sum_exp = tl.log(sum_val)
    tl.store(mid + pid, log_sum_exp)


@libentry()
@triton.jit
def logsumexp_kernel_2(mid, max_ptr, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    max_ptrs = max_ptr + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=0.0)
    max_val = tl.load(max_ptrs, mask=mask, other=-float("inf"))

    # For each block, we need to adjust by the global max
    # log(sum(exp(x - local_max))) = log(sum(exp(x - global_max)))
    #                               = local_log_sum_exp + (local_max - global_max)
    global_max = tl.max(max_val)
    adjustment = max_val - global_max
    adjusted = tl.exp(adjustment) * mid_val
    sum_val = tl.sum(adjusted, axis=0)
    log_sum_exp = tl.log(sum_val) + global_max
    tl.store(out, log_sum_exp)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def logsumexp_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    dtype = inp.type.element_ty
    neg_inf = tl.constexpr(-float("inf"))

    # Map the program id to the row of inp it should compute.
    pid = tle.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    acc_type = tl.float32
    _max = tl.full([BLOCK_M, BLOCK_N], value=neg_inf, dtype=acc_type)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask
        a = tl.load(inp + cols, mask, other=neg_inf)
        _max = tl.maximum(_max, a)
    row_max = tl.max(_max, axis=1)[:, None]

    # Second pass: compute sum(exp(x - row_max))
    _sum = tl.full([BLOCK_M, BLOCK_N], value=0.0, dtype=acc_type)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask
        a = tl.load(inp + cols, mask, other=neg_inf)
        _sum = _sum + tl.exp(a - row_max)
    row_sum = tl.sum(_sum, axis=1)[:, None]
    result = tl.log(row_sum) + row_max
    tl.store(out, result, row_mask)


def logsumexp(inp, dim=None, keepdim=False):
    logger.debug("GEMS_LOGSUMEXP")
    if dim is None or len(dim) == 0:
        # Reduce all dimensions
        M = inp.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        dtype = inp.dtype

        # Allocate intermediate buffers
        mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        max_buffer = torch.empty((mid_size,), dtype=dtype, device=inp.device)

        if not keepdim:
            out = torch.empty([], dtype=dtype, device=inp.device)
        else:
            shape = [1] * inp.dim()
            out = torch.empty(shape, dtype=dtype, device=inp.device)

        with torch_device_fn.device(inp.device):
            logsumexp_kernel_1[(mid_size, 1)](
                inp,
                mid,
                max_buffer,
                M,
                block_size,
            )
            logsumexp_kernel_2[(1, 1)](
                mid, max_buffer, out, mid_size, block_mid
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
            logsumexp_kernel[grid](inp, out, M, N)
        if not keepdim:
            out = out.squeeze(dim=dim)
        return out