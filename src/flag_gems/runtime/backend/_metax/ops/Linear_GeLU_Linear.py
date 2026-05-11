import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)

# Get cross-backend compatible functions
tanh = tl_extra_shim.tanh
pow = tl_extra_shim.pow


@libentry()
@triton.jit
def linear_gelu_linear_kernel_stage1(
    # Input and weights
    input_ptr,
    weight1_ptr,
    bias1_ptr,
    # Output intermediate
    intermediate_ptr,
    # Dimensions
    M,
    N1,
    K,
    # Strides
    stride_im,
    stride_ik,
    stride_w1k,
    stride_w1n,
    stride_b1n,
    stride_int_m,
    stride_int_n,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HAS_BIAS1: tl.constexpr,
):
    """
    Stage 1: Compute input @ weight1 = intermediate
    Then add bias1, apply GeLU
    """
    # Get program IDs
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

    # Calculate offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Matmul loop
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Calculate current K offset for this iteration
        k_offs = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        # Load input chunk
        i_ptrs = input_ptr + (offs_m[:, None] * stride_im + k_offs[None, :] * stride_ik)
        i_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x = tl.load(i_ptrs, mask=i_mask, other=0.0)

        # Load weight1 chunk
        w1_ptrs = weight1_ptr + (k_offs[:, None] * stride_w1k + offs_n[None, :] * stride_w1n)
        w1_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N1)
        w1 = tl.load(w1_ptrs, mask=w1_mask, other=0.0)

        # Accumulate
        accumulator += tl.dot(x, w1, allow_tf32=False)

    # Add bias1 if provided
    if HAS_BIAS1:
        b1_ptrs = bias1_ptr + offs_n * stride_b1n
        b1_mask = offs_n < N1
        bias1 = tl.load(b1_ptrs, mask=b1_mask, other=0.0)
        accumulator = accumulator + bias1

    # Apply GeLU activation
    gelu_out = 0.5 * accumulator * (1 + tanh(accumulator * 0.79788456 * (1 + 0.044715 * pow(accumulator, 2))))

    # Store intermediate result
    offs_int_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_int_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    int_ptrs = intermediate_ptr + (offs_int_m[:, None] * stride_int_m + offs_int_n[None, :] * stride_int_n)
    int_mask = (offs_int_m[:, None] < M) & (offs_int_n[None, :] < N1)
    tl.store(int_ptrs, gelu_out, mask=int_mask)


@libentry()
@triton.jit
def linear_gelu_linear_kernel_stage2(
    # Intermediate and weight2
    intermediate_ptr,
    weight2_ptr,
    bias2_ptr,
    # Output
    output_ptr,
    # Dimensions
    M,
    N1,
    N2,
    # Strides
    stride_int_m,
    stride_int_n,
    stride_w2k,
    stride_w2n,
    stride_b2n,
    stride_om,
    stride_on,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HAS_BIAS2: tl.constexpr,
):
    """
    Stage 2: Compute intermediate @ weight2 = output
    Then add bias2
    """
    # Get program IDs
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

    # Calculate offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Matmul loop
    for k in range(0, tl.cdiv(N1, BLOCK_SIZE_K)):
        # Calculate current N1 offset for this iteration
        k_offs = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        # Load intermediate chunk
        int_ptrs = intermediate_ptr + (offs_m[:, None] * stride_int_m + k_offs[None, :] * stride_int_n)
        int_mask = (offs_m[:, None] < M) & (k_offs[None, :] < N1)
        x = tl.load(int_ptrs, mask=int_mask, other=0.0)

        # Load weight2 chunk
        w2_ptrs = weight2_ptr + (k_offs[:, None] * stride_w2k + offs_n[None, :] * stride_w2n)
        w2_mask = (k_offs[:, None] < N1) & (offs_n[None, :] < N2)
        w2 = tl.load(w2_ptrs, mask=w2_mask, other=0.0)

        # Accumulate
        accumulator += tl.dot(x, w2, allow_tf32=False)

    # Add bias2 if provided
    if HAS_BIAS2:
        b2_ptrs = bias2_ptr + offs_n * stride_b2n
        b2_mask = offs_n < N2
        bias2 = tl.load(b2_ptrs, mask=b2_mask, other=0.0)
        accumulator = accumulator + bias2

    # Store output
    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = output_ptr + (offs_om[:, None] * stride_om + offs_on[None, :] * stride_on)
    output_mask = (offs_om[:, None] < M) & (offs_on[None, :] < N2)
    tl.store(output_ptrs, accumulator.to(tl.float32), mask=output_mask)


def linear_gelu_linear(
    input: torch.Tensor,
    weight1: torch.Tensor,
    bias1: torch.Tensor,
    weight2: torch.Tensor,
    bias2: torch.Tensor,
):
    """
    Fused Linear -> GeLU -> Linear operation.

    Args:
        input: Input tensor of shape (..., K)
        weight1: First weight matrix of shape (K, N1)
        bias1: First bias of shape (N1,), can be None
        weight2: Second weight matrix of shape (N1, N2)
        bias2: Second bias of shape (N2,), can be None

    Returns:
        Output tensor of shape (..., N2)
    """
    logger.debug("METAX GEMS LINEAR_GeLU_LINEAR")

    # Handle batch dimensions
    if input.dim() > 2:
        # Flatten batch dimensions
        batch_shape = input.shape[:-1]
        M = 1
        for dim in batch_shape:
            M *= dim
        K = input.shape[-1]
        input_flat = input.contiguous().view(M, K)
    elif input.dim() == 1:
        # Handle 1D input
        input_flat = input.unsqueeze(0).contiguous()
        M = 1
        K = input.shape[0]
        batch_shape = None
    else:
        batch_shape = None
        M, K = input.shape
        input_flat = input.contiguous()

    N1 = weight1.shape[1]
    N2 = weight2.shape[1]

    # Ensure weights and biases are contiguous
    weight1 = weight1.contiguous()
    weight2 = weight2.contiguous()

    # Handle biases - create empty tensors if None
    has_bias1 = bias1 is not None
    has_bias2 = bias2 is not None

    if has_bias1:
        bias1 = bias1.contiguous()
    else:
        bias1 = torch.empty(1, device=input.device, dtype=input.dtype)

    if has_bias2:
        bias2 = bias2.contiguous()
    else:
        bias2 = torch.empty(1, device=input.device, dtype=input.dtype)

    # Stage 1: Compute intermediate = input @ weight1 + bias1, then apply GeLU
    intermediate = torch.empty((M, N1), device=input.device, dtype=input.dtype)

    grid1 = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N1, META["BLOCK_SIZE_N"]),
    )

    with torch_device_fn.device(input.device):
        linear_gelu_linear_kernel_stage1[grid1](
            input_flat,
            weight1,
            bias1,
            intermediate,
            M,
            N1,
            K,
            input_flat.stride(0),
            input_flat.stride(1),
            weight1.stride(0),
            weight1.stride(1),
            bias1.stride(0) if has_bias1 else 0,
            intermediate.stride(0),
            intermediate.stride(1),
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=32,
            HAS_BIAS1=has_bias1,
        )

    # Stage 2: Compute output = intermediate @ weight2 + bias2
    output = torch.empty((M, N2), device=input.device, dtype=input.dtype)

    grid2 = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N2, META["BLOCK_SIZE_N"]),
    )

    with torch_device_fn.device(input.device):
        linear_gelu_linear_kernel_stage2[grid2](
            intermediate,
            weight2,
            bias2,
            output,
            M,
            N1,
            N2,
            intermediate.stride(0),
            intermediate.stride(1),
            weight2.stride(0),
            weight2.stride(1),
            bias2.stride(0) if has_bias2 else 0,
            output.stride(0),
            output.stride(1),
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=32,
            HAS_BIAS2=has_bias2,
        )

    # Reshape output to match input batch shape
    if batch_shape is not None:
        output = output.view(*batch_shape, N2)
    elif input.dim() == 1:
        output = output.squeeze(0)

    return output