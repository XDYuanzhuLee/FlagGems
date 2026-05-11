import logging
import math
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

rsqrt = tl_extra_shim.rsqrt
tanh = tl_extra_shim.tanh

logger = logging.getLogger(__name__)


@triton.jit
def layer_norm_gelu_kernel(
    input_ptr,
    output_ptr,
    weight_ptr,
    bias_ptr,
    M,
    N,
    eps,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused LayerNorm + GeLU kernel.
    Applies Layer Normalization to the last N dimensions, then applies GeLU.
    """
    pid_m = tl.program_id(0)

    # Offsets for the row
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    # Mask for valid elements
    mask_m = offs_m < M
    mask = mask_m[:, None] & (offs_n[None, :] < N)

    # Load input data
    input_ptrs = (
        input_ptr + offs_m[:, None] * stride_in_m + offs_n[None, :] * stride_in_n
    )
    x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Compute mean
    mean = tl.sum(x, axis=1) / N
    # Compute variance
    d = x - mean[:, None]
    var = tl.sum(d * d, axis=1) / N
    rstd = rsqrt(var + eps)

    # Normalize
    x_hat = d * rstd[:, None]

    # Apply weight and bias if provided
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + offs_n, mask=offs_n < N)
        x_hat = x_hat * weight[None, :]

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N)
        x_hat = x_hat + bias[None, :]

    # Apply GeLU (tanh approximation)
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x^3)))
    gelu_out = 0.5 * x_hat * (1 + tanh(0.79788456 * x_hat * (1 + 0.044715 * x_hat * x_hat)))

    # Store output
    output_ptrs = (
        output_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    )
    tl.store(output_ptrs, gelu_out.to(x.dtype), mask=mask)


def layer_norm_gelu(
    input_tensor: torch.Tensor,
    normalized_shape: list,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Fused LayerNorm + GeLU operation.

    Args:
        input_tensor: Input tensor
        normalized_shape: Shape to normalize over (typically the last dimensions)
        weight: Optional weight for LayerNorm
        bias: Optional bias for LayerNorm
        eps: Epsilon for numerical stability

    Returns:
        Output tensor after LayerNorm + GeLU
    """
    logger.debug("GEMS LAYER_NORM_GELU")

    # Handle different input shapes
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]

    # Calculate dimensions
    N = math.prod(normalized_shape)
    M = input_tensor.numel() // N

    # Ensure contiguous
    input_tensor = input_tensor.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()

    # Output shape matches input shape
    output = torch.empty_like(input_tensor)

    # Grid configuration
    BLOCK_SIZE_M = 1
    BLOCK_SIZE_N = triton.next_power_of_2(min(N, 4096))

    grid = (M,)

    layer_norm_gelu_kernel[grid](
        input_tensor,
        output,
        weight,
        bias,
        M,
        N,
        eps,
        input_tensor.stride(0) if M > 1 else 0,
        input_tensor.stride(-1),
        output.stride(0) if M > 1 else 0,
        output.stride(-1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return output