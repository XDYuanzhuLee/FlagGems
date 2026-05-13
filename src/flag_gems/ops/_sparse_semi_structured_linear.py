import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger("flag_gems._sparse_semi_structured_linear")


@libentry()
@triton.jit
def _sparse_linear_kernel(
    input_ptr,
    weight_ptr,
    meta_ptr,
    output_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_im,
    stride_ik,
    stride_wk,
    stride_wn,
    stride_mm,
    stride_mn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Kernel for sparse semi-structured linear layer.

    The 2:4 sparse format stores 4 elements per group with 2 non-zeros.
    The meta tensor encodes which elements are valid.
    """
    pid = tl.program_id(0)
    pid_m = pid // tl.cdiv(N, BLOCK_N)
    pid_n = pid % tl.cdiv(N, BLOCK_N)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Load input block
    input_mask = (offs_m < M)[:, None] & (offs_k < K)[None, :]
    input_ptrs = input_ptr + (offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik)
    input_block = tl.load(input_ptrs, mask=input_mask, other=0.0)

    # Compute output block using sparse weight
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load weight block
        weight_mask = (offs_k[:, None] < K) & (offs_n < N)[None, :]
        weight_ptrs = weight_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        weight_block = tl.load(weight_ptrs, mask=weight_mask, other=0.0)

        # Load meta block for this K slice
        meta_ptrs = meta_ptr + (offs_k * stride_mm)
        meta_block = tl.load(meta_ptrs, mask=(offs_k < K), other=0)

        # Apply mask: keep elements where meta is non-zero
        # Meta format: each 4 elements has a 2-bit mask
        masked_weight = tl.where(meta_block != 0, weight_block, 0.0)

        # Matrix multiply accumulate
        accumulator += tl.dot(input_block, masked_weight, allow_tf32=False)

        offs_k += BLOCK_K
        input_ptrs += BLOCK_K * stride_ik
        weight_ptrs += BLOCK_K * stride_wk
        meta_ptrs += BLOCK_K * stride_mm

    # Add bias if provided
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_n)
        accumulator += bias

    # Store result
    output_mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    output_ptrs = output_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(output_ptrs, accumulator, mask=output_mask)


def _sparse_semi_structured_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    meta: torch.Tensor,
    bias: torch.Tensor = None,
    activation: str = None,
    out_dtype: torch.dtype = None,
):
    """
    Implements sparse semi-structured linear layer.

    The 2:4 sparse format stores metadata indicating which elements are valid.
    This implementation uses the metadata to mask out invalid elements during computation.
    """
    logger.debug("GEMS SPARSE SEMI STRUCTURED LINEAR")

    M, K = input.shape
    N = weight.shape[0]
    K_w = weight.shape[1]

    assert K == K_w, f"Incompatible dimensions: input K={K}, weight K={K_w}"

    # Determine output dtype
    if out_dtype is not None:
        output_dtype = out_dtype
    else:
        output_dtype = input.dtype

    # Allocate output
    output = torch.empty((M, N), device=input.device, dtype=output_dtype)

    # Determine block sizes
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16

    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
    )

    _sparse_linear_kernel[grid](
        input,
        weight,
        meta,
        output,
        bias if bias is not None else 0,  # Pass 0 if no bias for kernel
        M,
        N,
        K,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        meta.stride(0),
        meta.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Apply activation if specified
    if activation is not None:
        if activation == "relu":
            output = torch.relu(output)
        elif activation == "silu" or activation == "swish":
            output = torch.silu(output)
        elif activation == "gelu":
            output = torch.gelu(output)
        else:
            logger.warning(f"Unknown activation: {activation}")

    return output