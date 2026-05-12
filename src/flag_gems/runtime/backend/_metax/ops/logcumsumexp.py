import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)


@libentry()
@triton.jit
def logcumsumexp_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_N: tl.constexpr,
):
    """Kernel for logcumsumexp along the last dimension (row scan)."""
    pid_m = tle.program_id(0)
    row_offset = pid_m * N

    # Load the row and compute in float32
    col_offsets = tl.arange(0, BLOCK_N)
    mask = col_offsets < N

    # Load input values as float32
    inp_ptrs = inp + row_offset + col_offsets
    inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf")).to(tl.float32)

    # Compute exp
    exp_vals = tl.exp(inp_vals)

    # Compute cumulative sum of exp
    cumsum_exp = tl.cumsum(exp_vals, axis=0)

    # Compute log
    log_cumsum_exp = tl.log(cumsum_exp)

    # Store result
    out_ptrs = out + row_offset + col_offsets
    tl.store(out_ptrs, log_cumsum_exp, mask=mask)


@libentry()
@triton.jit
def logcumsumexp_kernel_3d(
    inp,
    out,
    M,
    N,
    K,
    BLOCK_N: tl.constexpr,
):
    """Kernel for logcumsumexp for 3D+ tensors (general case)."""
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)

    row_offset = pid_m * N * K
    k_offset = pid_k

    # Load the row and compute in float32
    col_offsets = tl.arange(0, BLOCK_N)
    mask = col_offsets < N

    # Load input values as float32
    inp_ptrs = inp + row_offset + col_offsets * K + k_offset
    inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf")).to(tl.float32)

    # Compute exp
    exp_vals = tl.exp(inp_vals)

    # Compute cumulative sum of exp
    cumsum_exp = tl.cumsum(exp_vals, axis=0)

    # Compute log
    log_cumsum_exp = tl.log(cumsum_exp)

    # Store result
    out_ptrs = out + row_offset + col_offsets * K + k_offset
    tl.store(out_ptrs, log_cumsum_exp, mask=mask)


def heur_block_n(args):
    return triton.next_power_of_2(args["N"])


def logcumsumexp(inp, dim):
    logger.debug("METAX GEMS LOGCUMSUMEXP")

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim
    shape = list(inp.shape)

    # Compute M (product of dims before dim), N (dim size), K (product of dims after dim)
    N = shape[dim]
    M = 1
    for i in range(dim):
        M *= shape[i]
    K = inp.numel() // M // N

    # Ensure contiguous input
    inp = inp.contiguous()

    # Output shape is same as input
    out = torch.empty_like(inp, dtype=torch.float32)

    if K == 1:
        # Simple case: row scan
        BLOCK_N = min(triton.next_power_of_2(N), 4096)
        grid = (M,)
        with torch_device_fn.device(inp.device):
            logcumsumexp_kernel[grid](
                inp,
                out,
                M,
                N,
                BLOCK_N,
            )
    else:
        # General case: 3D+ tensor
        BLOCK_N = min(triton.next_power_of_2(N), 4096)
        grid = (M, K)
        with torch_device_fn.device(inp.device):
            logcumsumexp_kernel_3d[grid](
                inp,
                out,
                M,
                N,
                K,
                BLOCK_N,
            )

    # Convert back to original dtype if needed
    if inp.dtype != torch.float32:
        out = out.to(inp.dtype)

    return out