import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def logsumexp_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M

    inp_val = tl.load(inp_ptrs, mask=mask, other=float("-inf"))
    # Numerical stabilization: subtract max
    row_max = tl.max(inp_val)
    inp_shifted = inp_val - row_max
    exp_inp = tl.exp(inp_shifted)
    sum_exp = tl.sum(exp_inp)
    log_sum_exp = tl.log(sum_exp) + row_max

    mid_ptr = mid + pid
    tl.store(mid_ptr, log_sum_exp)


@libentry()
@triton.jit
def logsumexp_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size

    mid_val = tl.load(mid_ptrs, mask=mask, other=float("-inf"))
    # Numerical stabilization for second level
    row_max = tl.max(mid_val)
    mid_shifted = mid_val - row_max
    exp_mid = tl.exp(mid_shifted)
    sum_exp = tl.sum(exp_mid)
    log_sum_exp = tl.log(sum_exp) + row_max

    tl.store(out, log_sum_exp)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("logsumexp"))
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
    pid = tle.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    # Numerical stabilization: compute row max first
    _all = tl.full([BLOCK_M, BLOCK_N], value=float("-inf"), dtype=dtype)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask
        a = tl.load(inp + cols, mask, other=float("-inf"))
        _all = tl.maximum(_all, a)

    row_max = tl.max(_all, axis=1)[:, None]

    # Second pass: compute sum of exp(x - max)
    _all = tl.full([BLOCK_M, BLOCK_N], value=float("-inf"), dtype=dtype)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask
        a = tl.load(inp + cols, mask, other=float("-inf"))
        shifted = a - row_max
        exp_val = tl.exp(shifted)
        _all = tl.exp(tl.where(mask, shifted, float("-inf"))) + tl.where(mask, _all, 0)
        _all = tl.where(mask, _all + exp_val, _all)

    # Actually, let's use a cleaner approach
    sum_exp = tl.sum(tl.exp(_all - row_max), axis=1)[:, None]
    result = tl.log(sum_exp) + row_max

    tl.store(out, result, row_mask)


def logsumexp(inp, dim, keepdim=False):
    logger.debug("GEMS LOGSUMEXP")
    # Convert dim to a list if it's an int or None
    if dim is None:
        dim = []
    elif isinstance(dim, int):
        dim = [dim]
    elif isinstance(dim, (tuple, list)):
        dim = list(dim)
    else:
        dim = [dim]

    if len(dim) == 0:
        # Reduce all dimensions
        M = inp.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        dtype = inp.dtype
        mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        if not keepdim:
            out = torch.empty([], dtype=dtype, device=inp.device)
        else:
            shape = [1] * inp.dim()
            out = torch.empty(shape, dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            logsumexp_kernel_1[(mid_size, 1)](
                inp,
                mid,
                M,
                block_size,
            )
            logsumexp_kernel_2[(1, 1)](
                mid, out, mid_size, block_mid
            )
        return out
    else:
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