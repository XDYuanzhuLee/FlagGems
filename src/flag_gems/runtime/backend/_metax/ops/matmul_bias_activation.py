import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)


# Block sizes for matmul
BLOCK_SIZE_M = 32
BLOCK_SIZE_N = 64
BLOCK_SIZE_K = 32


@libentry()
@triton.jit
def matmul_bias_activation_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # Load bias - bias is 1D of shape (N,), so we only index by column (offs_cn)
    bias_ptrs = bias_ptr + offs_cn
    bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0)

    # Add bias (broadcast 1D bias to 2D)
    accumulator = accumulator + bias

    # Apply ReLU activation: max(0, x)
    accumulator = tl.where(accumulator > 0, accumulator, 0.0)

    c = accumulator.to(bias.dtype)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_bias_activation(input, weight, bias):
    """
    Fused matmul + bias + ReLU activation.

    Args:
        input: Input tensor of shape (..., K) or (M, K)
        weight: Weight tensor of shape (N, K)
        bias: Bias tensor of shape (N,)

    Returns:
        Output tensor of shape (..., N) or (M, N)
    """
    logger.debug("METAX GEMS Matmul_Bias_Activation")

    # Handle broadcasting for input
    if input.dim() == 1:
        input = input.unsqueeze(0)
        squeeze_output = True
    elif input.dim() > 2:
        original_shape = input.shape
        input = input.view(-1, input.shape[-1])
        squeeze_output = False
    else:
        squeeze_output = False

    # Get dimensions
    M, K = input.shape
    N = weight.shape[0]

    assert weight.shape == (N, K), "Incompatible dimensions"
    assert broadcastable_to(bias.shape, (N,)), "Incompatible bias shape"

    logger.debug(
        "METAX GEMS Matmul_Bias_Activation, [shape info]: M=%s, N=%s, K=%s",
        M, N, K
    )

    input = input.contiguous()
    weight = weight.t().contiguous()  # weight.T to make it (K, N)
    out = torch.empty((M, N), device=input.device, dtype=input.dtype)
    bias = bias.contiguous()

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    with torch_device_fn.device(input.device):
        matmul_bias_activation_kernel[grid](
            input,
            weight,
            bias,
            out,
            M,
            N,
            K,
            input.stride(0),
            input.stride(1),
            weight.stride(0),
            weight.stride(1),
            bias.stride(0),
            bias.stride(1) if bias.dim() > 1 else 0,
            out.stride(0),
            out.stride(1),
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
        )

    # Reshape output if needed
    if squeeze_output:
        out = out.squeeze(0)
    elif input.dim() > 2:
        out = out.view(*original_shape[:-1], N)

    return out