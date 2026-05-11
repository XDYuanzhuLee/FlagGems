import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.zeros import zero_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)


def heur_block_n(args):
    return triton.next_power_of_2(args["N"])


def heur_num_warps(args):
    if args["N"] <= 1024:
        return 1
    elif args["N"] <= 2048:
        return 4
    else:
        return 8


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("log_softmax"), key=["M", "N"])
@triton.heuristics(
    {
        "BLOCK_N": heur_block_n,
        "num_warps": heur_num_warps,
    }
)
@triton.jit
def fused_softmax_kernel(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_K: tl.constexpr,
):
    """Fused softmax kernel for Metax backend.

    Computes: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    """
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)

    # Offset calculations
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)

    # Calculate offset based on whether we're using K dimension
    if USE_K:
        row_offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
    else:
        row_offset = m_offset[:, None] * N + n_offset[None, :]

    mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
    input_ptrs = input_ptr + row_offset

    # Load input
    inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)

    # Compute softmax: exp(x - max) / sum(exp(x - max))
    row_max = tl.max(inp, axis=1)[:, None]
    row_minus_max = inp - row_max
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1)[:, None]
    softmax_output = numerator / denominator

    # Store output
    output_ptrs = output_ptr + row_offset
    tl.store(output_ptrs, softmax_output, mask=mask)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("log_softmax"), key=["M", "N"])
@triton.heuristics(
    {
        "BLOCK_N": heur_block_n,
        "num_warps": heur_num_warps,
    }
)
@triton.jit
def fused_softmax_backward_kernel(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_K: tl.constexpr,
):
    """Backward kernel for fused softmax."""
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)

    # Offset calculations
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)

    # Calculate offset
    if USE_K:
        row_col_offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
    else:
        row_col_offset = m_offset[:, None] * N + n_offset[None, :]

    mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)

    # Load output and gradient
    out_tile = tl.load(out_ptr + row_col_offset, mask=mask).to(tl.float32)
    out_grad_tile = tl.load(out_grad_ptr + row_col_offset, mask=mask).to(tl.float32)

    # Compute scale factor: sum(out * out_grad)
    scale = tl.sum(out_tile * out_grad_tile, axis=1)[:, None]

    # Compute input gradient: out * (out_grad - scale)
    in_grad_tile = out_tile * (out_grad_tile - scale)

    # Store input gradient
    in_grad_ptrs = in_grad_ptr + row_col_offset
    tl.store(in_grad_ptrs, in_grad_tile, mask=mask)


def fused_softmax(self, dim, half_to_float=False):
    """Fused softmax implementation for Metax backend.

    This is an optimized softmax implementation that computes:
    softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    Args:
        self: Input tensor
        dim: Dimension along which to compute softmax
        half_to_float: If True, cast input from half to float precision

    Returns:
        Softmax output tensor
    """
    logger.debug("METAX GEMS FUSED_SOFTMAX")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"

    # Special handling for empty tensor
    if self.numel() == 0:
        out_shape = list(self.shape)
        out = torch.empty(out_shape, dtype=self.dtype, device=self.device)
        zero_(out)
        return out

    dim = dim % self.ndim
    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]

    inp = self.contiguous()
    if half_to_float:
        dtype = torch.float32
    else:
        dtype = self.dtype
    out = torch.empty_like(inp, dtype=dtype)
    K = inp.numel() // M // N
    USE_K = K != 1

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    with torch_device_fn.device(inp.device):
        fused_softmax_kernel[grid](
            out,
            inp,
            M,
            N,
            K,
            USE_K=USE_K,
        )

    return out


def fused_softmax_backward(grad_output, output, dim, input_dtype):
    """Backward pass for fused softmax.

    Args:
        grad_output: Gradient of the loss w.r.t. softmax output
        output: Softmax output from forward pass
        dim: Dimension along which softmax was computed
        input_dtype: Original input dtype

    Returns:
        Gradient w.r.t. input
    """
    logger.debug("METAX GEMS FUSED_SOFTMAX VJP")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    grad_output = grad_output.contiguous()
    in_grad = torch.empty_like(output, dtype=input_dtype)
    K = output.numel() // M // N
    USE_K = K != 1

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    with torch_device_fn.device(in_grad.device):
        fused_softmax_backward_kernel[grid](
            output,
            grad_output,
            in_grad,
            M,
            N,
            K,
            USE_K=USE_K,
        )

    return in_grad