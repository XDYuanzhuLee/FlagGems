import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _weight_int4pack_mm_with_scales_and_zeros_kernel(
    A,  # activation: (M, K), float16
    B,  # weight: (N, K/2), uint8 packed int4
    C,  # output: (M, N)
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    qGroupSize,
    qScale,  # (N, K/qGroupSize)
    qZero,   # (N, K/qGroupSize)
    stride_qs,
    stride_qz,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Weight-only int4 matrix multiplication kernel with separate scales and zeros."""
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M).to(tl.int64)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N).to(tl.int64)
    rm = rm.to(tl.int64)
    rn = rn.to(tl.int64)

    # Initialize accumulator with float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Process K dimension in chunks
    for start_k in range(0, K, BLOCK_K):
        rk = (start_k + tl.arange(0, BLOCK_K)).to(tl.int64)
        mask_k = rk < K

        # Load activation A: (M, K)
        a_ptrs = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0).to(tl.float32)

        # Load weight B: (N, K/2) packed int4
        rk_packed = (start_k + tl.arange(0, BLOCK_K)) // 2
        rk_packed = rk_packed.to(tl.int64)
        b_ptrs = B + (rbn[:, None] * stride_bn + rk_packed[None, :] * stride_bk)
        b_raw = tl.load(b_ptrs, mask=rk_packed[None, :] < (K // 2), other=0)
        b_as_int = b_raw.to(tl.int32)

        # Unpack int4 from uint8
        b_int4_low = (b_as_int & 0x0F).to(tl.float32) - 8.0
        b_int4_high = ((b_as_int >> 4) & 0x0F).to(tl.float32) - 8.0

        # Get group index
        group_idx = (start_k + tl.arange(0, BLOCK_K)) // qGroupSize
        group_idx = group_idx.to(tl.int64)

        # Load scales and zeros for the groups
        qs_ptrs = qScale + (rbn[:, None] * stride_qs + group_idx[None, :])
        qz_ptrs = qZero + (rbn[:, None] * stride_qz + group_idx[None, :])
        scale = tl.load(qs_ptrs, mask=group_idx[None, :] < (K // qGroupSize), other=1.0).to(tl.float32)
        zero = tl.load(qz_ptrs, mask=group_idx[None, :] < (K // qGroupSize), other=0.0).to(tl.float32)

        # Dequantize: (value - zero) * scale
        b_dequant_low = (b_int4_low - zero) * scale
        b_dequant_high = (b_int4_high - zero) * scale

        # Interleave low and high nibbles
        b_dequant = tl.where(
            (rk % 2 == 0)[None, :],
            b_dequant_low,
            b_dequant_high
        )

        # Matrix multiplication: transpose b_dequant and use tl.dot
        acc += tl.dot(a, tl.trans(b_dequant), out_dtype=tl.float32, allow_tf32=False)

    # Store result directly without type conversion
    c_ptrs = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def _weight_int4pack_mm_with_scales_and_zeros(act, weight, qGroupSize, qScale, qZero):
    """Weight-only int4 matrix multiplication with separate scales and zeros."""
    M, K = act.shape
    N = weight.shape[0]

    logger.debug(
        "ILUVATAR GEMS WEIGHT_INT4PACK_MM_WITH_SCALES_AND_ZEROS, M=%s, N=%s, K=%s, qGroupSize=%s",
        M, N, K, qGroupSize
    )

    # Use float32 output and convert to target dtype later
    output = torch.empty((M, N), dtype=torch.float32, device=act.device)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    GROUP_M = 8

    grid = lambda META: (
        triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
    )

    with torch_device_fn.device(act.device):
        _weight_int4pack_mm_with_scales_and_zeros_kernel[grid](
            act,
            weight,
            output,
            M, N, K,
            act.stride(0), act.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
            qGroupSize,
            qScale,
            qZero,
            qScale.stride(0),
            qZero.stride(0),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M,
        )

    # Convert to target dtype
    return output.to(act.dtype)