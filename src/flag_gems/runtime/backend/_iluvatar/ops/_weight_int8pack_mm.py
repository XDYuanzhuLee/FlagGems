import logging
from typing import List

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _weight_int8pack_mm_kernel(
    A,
    B,
    scales,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_sm,
    stride_sn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=offs_k[None, :] < K - k * BLOCK_SIZE_K,
            other=0,
            eviction_policy="evict_last",
        )
        b = tl.load(
            b_ptrs,
            mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
            other=0,
            eviction_policy="evict_last",
        )

        # Triton dot requires int8/uint8 input, accumulate in int32
        accumulator += tl.dot(a, b, out_dtype=tl.int32)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_sm = offs_am
    offs_sn = offs_bn
    scale_ptrs = scales + offs_sm[:, None] * stride_sm + offs_sn[None, :] * stride_sn
    scale_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    scales_val = tl.load(scale_ptrs, mask=scale_mask, other=0.0)

    accumulator = accumulator * scales_val

    if C.dtype == tl.float16:
        c = accumulator.to(tl.float16)
    elif C.dtype == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


configs = [
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 4},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 4},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 4},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 4},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 4},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 4},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 4},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 4},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 4},
        num_warps=8,
        num_stages=2,
    ),
]


@libentry()
@triton.autotune(configs=configs, key=["M", "N", "K"])
@triton.jit
def _weight_int8pack_mm_kernel_autotuned(
    A,
    B,
    scales,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_sm,
    stride_sn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=offs_k[None, :] < K - k * BLOCK_SIZE_K,
            other=0,
            eviction_policy="evict_last",
        )
        b = tl.load(
            b_ptrs,
            mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
            other=0,
            eviction_policy="evict_last",
        )

        # Triton dot requires int8/uint8 input, accumulate in int32
        accumulator += tl.dot(a, b, out_dtype=tl.int32)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_sm = offs_am
    offs_sn = offs_bn
    scale_ptrs = scales + offs_sm[:, None] * stride_sm + offs_sn[None, :] * stride_sn
    scale_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    scales_val = tl.load(scale_ptrs, mask=scale_mask, other=0.0)

    accumulator = accumulator * scales_val

    if C.dtype == tl.float16:
        c = accumulator.to(tl.float16)
    elif C.dtype == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def _weight_int8pack_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Performs quantized matrix multiplication using int8 inputs with per-tensor scales.

    Args:
        A: int8 tensor of shape [M, K]
        B: int8 tensor of shape [N, K] (packed weights)
        scales: float16/float32 tensor of shape [M, N]
        output_dtype: output dtype (default: float16)

    Returns:
        Output tensor of shape [M, N]
    """
    logger.debug("ILUVATAR GEMS _weight_int8pack_mm")

    assert A.dtype == torch.int8, f"Expected int8 for A, got {A.dtype}"
    assert B.dtype == torch.int8, f"Expected int8 for B, got {B.dtype}"
    assert scales.dtype in [torch.float16, torch.float32, torch.bfloat16], (
        f"Expected float16/float32/bfloat16 for scales, got {scales.dtype}"
    )

    M, K = A.shape
    N, K_b = B.shape
    assert K == K_b, f"K dimension mismatch: A has {K}, B has {K_b}"

    assert scales.shape == (M, N), f"Scales shape mismatch: expected ({M}, {N}), got {scales.shape}"

    # Create output tensor
    C = A.new_empty((M, N), dtype=output_dtype)

    # Handle empty tensors
    if M == 0 or N == 0 or K == 0:
        return C

    def grid(META):
        return (
            triton.cdiv(M, 64)
            * triton.cdiv(N, 64),
        )

    _weight_int8pack_mm_kernel[grid](
        A,
        B,
        scales,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(1),
        B.stride(0),
        C.stride(0),
        C.stride(1),
        scales.stride(0),
        scales.stride(1),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=128,
        GROUP_SIZE_M=4,
    )

    return C