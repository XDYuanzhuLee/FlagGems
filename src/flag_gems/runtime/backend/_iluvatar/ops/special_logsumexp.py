import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def logsumexp_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_N: tl.constexpr,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    pid = tle.program_id(0)
    if pid >= M:
        return

    row_start = pid * N
    inp = inp + row_start

    max_val = tl.full([BLOCK_N], value=float("-inf"), dtype=cdtype)
    sum_val = tl.full([BLOCK_N], value=0.0, dtype=cdtype)

    for off in range(0, N, BLOCK_N):
        cols = tl.arange(0, BLOCK_N)
        col_mask = cols < N
        mask = col_mask

        a = tl.load(inp + cols, mask=col_mask, other=0).to(cdtype)

        new_max = tl.maximum(max_val, a)
        sum_val = tl.where(mask, sum_val * tl.exp(max_val - new_max) + tl.exp(a - new_max), sum_val)
        max_val = new_max

    final_max = tl.max(max_val)
    final_sum = tl.sum(sum_val * tl.exp(max_val - final_max))
    result = tl.log(final_sum) + final_max

    tl.store(out + pid, result)


def logsumexp(inp, dim, keepdim=False):
    logger.debug("ILUVATAR GEMS LOGSUMEXP")
    M = inp.numel() // inp.shape[dim]
    N = inp.shape[dim]

    dim = dim if dim >= 0 else dim + inp.ndim

    out_shape = list(inp.shape)
    if keepdim:
        out_shape[dim] = 1
    else:
        out_shape = out_shape[:dim] + out_shape[dim + 1 :]

    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    inp = inp.contiguous()

    BLOCK_N = 1024

    with torch_device_fn.device(inp.device):
        grid = lambda meta: (M,)
        logsumexp_kernel[grid](inp, out, M, N, BLOCK_N)

    return out