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
def reduce_norm_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tle.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Out = Out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _sum += a * a
    sum = tl.sum(_sum, axis=1)

    out = tl.sqrt(sum)[:, None]
    tl.store(Out, out, row_mask)


@libentry()
@triton.jit
def reduce_norm_kernel_1(X, Mid, M, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0).to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    X = X + offset
    Mid = Mid + pid
    mask = offset < M

    x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
    mid = tl.sum(x * x)
    tl.store(Mid, mid)


@libentry()
@triton.jit
def reduce_norm_kernel_2(Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    Mid = Mid + offset
    mask = offset < MID_SIZE
    mid = tl.load(Mid, mask=mask, other=0.0).to(tl.float32)
    out = tl.sqrt(tl.sum(mid))
    tl.store(Out, out)


def reduce_norm(x, dim=None, keepdim=False):
    """
    Compute the L2 norm (Euclidean norm) of the input tensor.

    This is equivalent to torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=keepdim)

    Args:
        x: Input tensor
        dim: Dimension(s) along which to compute the norm. If None, computes norm over all elements.
        keepdim: If True, the output tensor has dim retained as dimension of size 1.

    Returns:
        The L2 norm of the input tensor.
    """
    logger.debug("GEMS REDUCE NORM")
    dtype = x.dtype
    if dtype not in [torch.float16, torch.float32, torch.bfloat16]:
        raise NotImplementedError(f"reduce_norm not implemented for {dtype}")

    with torch_device_fn.device(x.device):
        if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == x.ndim):
            # Compute norm over all elements (flatten)
            dim = list(range(x.ndim))
            shape = [1] * x.ndim
            x = dim_compress(x, dim)
            M = x.numel()
            BLOCK_SIZE = triton.next_power_of_2(math.ceil(math.sqrt(M)))
            MID_SIZE = triton.cdiv(M, BLOCK_SIZE)
            BLOCK_MID = triton.next_power_of_2(MID_SIZE)

            mid = torch.empty([MID_SIZE], dtype=torch.float32, device=x.device)
            out = torch.empty(shape, dtype=dtype, device=x.device)

            reduce_norm_kernel_1[(MID_SIZE,)](x, mid, M, BLOCK_SIZE)
            reduce_norm_kernel_2[(1,)](mid, out, MID_SIZE, BLOCK_MID)
        else:
            # Compute norm over specified dimensions
            shape = list(x.shape)
            if isinstance(dim, int):
                dim = [dim]
            dim = [d % x.ndim for d in dim]
            x = dim_compress(x, dim)
            N = 1
            for i in dim:
                N *= shape[i]
                shape[i] = 1
            M = x.numel() // N
            out = torch.empty(shape, dtype=dtype, device=x.device)
            grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
            reduce_norm_kernel[grid](x, out, M, N)

    if not keepdim:
        if dim is None:
            # For scalar output
            out = out.squeeze()
        else:
            out = out.squeeze(dim=dim)
    return out