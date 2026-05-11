import logging
import math
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle

rsqrt = tl_extra_shim.rsqrt
tanh = tl_extra_shim.tanh

logger = logging.getLogger("flag_gems." + __name__)


def heur_block_n(args):
    return triton.next_power_of_2(min(args["N"], 4096))


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("layer_norm_gelu"),
    key=["M", "N"],
)
@triton.heuristics(
    {
        "BLOCK_N": heur_block_n,
    }
)
@triton.jit(do_not_specialize=["eps"])
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
    BLOCK_N: tl.constexpr,
):
    """
    Fused LayerNorm + GeLU kernel for Metax backend.
    Applies Layer Normalization to the last N dimensions, then applies GeLU.
    """
    pid_m = tle.program_id(0)

    # Offsets for the row
    offs_n = tl.arange(0, BLOCK_N)

    # Mask for valid elements
    mask = offs_n < N

    # Load input data
    input_ptrs = input_ptr + pid_m * stride_in_m + offs_n * stride_in_n
    x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Compute mean
    mean = tl.sum(x, axis=0) / N
    # Compute variance
    d = x - mean
    var = tl.sum(d * d, axis=0) / N
    rstd = rsqrt(var + eps)

    # Normalize
    x_hat = d * rstd

    # Apply weight and bias if provided
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + offs_n, mask=mask)
        x_hat = x_hat * weight

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_n, mask=mask)
        x_hat = x_hat + bias

    # Apply GeLU (tanh approximation)
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x^3)))
    gelu_out = 0.5 * x_hat * (1 + tanh(0.79788456 * x_hat * (1 + 0.044715 * x_hat * x_hat)))

    # Store output
    output_ptrs = output_ptr + pid_m * stride_out_m + offs_n * stride_out_n
    tl.store(output_ptrs, gelu_out.to(x.dtype), mask=mask)


def layer_norm_gelu(
    input_tensor: torch.Tensor,
    normalized_shape: list,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Fused LayerNorm + GeLU operation for Metax backend.

    Args:
        input_tensor: Input tensor
        normalized_shape: Shape to normalize over (typically the last dimensions)
        weight: Optional weight for LayerNorm
        bias: Optional bias for LayerNorm
        eps: Epsilon for numerical stability

    Returns:
        Output tensor after LayerNorm + GeLU
    """
    logger.debug("METAX GEMS LAYER_NORM_GELU")

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

    # For efficient access, reshape input and output to (M, N) if needed
    # This makes stride_in_m = N (contiguous)
    input_shape = input_tensor.shape
    num_non_normalized_dims = len(input_shape) - len(normalized_shape)

    if num_non_normalized_dims > 1:
        # Multi-dimensional case: need to reshape to 2D for proper access
        # Flatten the non-normalized dimensions into M
        input_2d = input_tensor.view(M, N)
        output_2d = output.view(M, N)
        stride_in_m = N
        stride_out_m = N
    else:
        # Simple 2D case: input is already (M, N)
        input_2d = input_tensor
        output_2d = output
        stride_in_m = input_tensor.stride(0) if M > 1 else 0
        stride_out_m = output.stride(0) if M > 1 else 0

    stride_in_n = 1
    stride_out_n = 1

    # Grid configuration
    grid = (M,)

    with torch_device_fn.device(input_tensor.device):
        layer_norm_gelu_kernel[grid](
            input_2d,
            output_2d,
            weight,
            bias,
            M,
            N,
            eps,
            stride_in_m,
            stride_in_n,
            stride_out_m,
            stride_out_n,
        )

    return output