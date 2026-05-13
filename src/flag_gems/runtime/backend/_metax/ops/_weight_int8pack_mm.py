import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)


@libentry()
@triton.heuristics(
    {
        "USE_LARGE_BLOCK": lambda args: args["M"] * args["N"] >= 8192,
    }
)
@triton.jit
def weight_int8pack_mm_kernel(
    weight_ptr,  # packed int8 weight (M, K//2)
    mat2_ptr,    # activation (K, N)
    scales_ptr,  # scales (M,)
    output_ptr,  # output (M, N)
    M,
    N,
    K,
    stride_weight_m,
    stride_weight_k,
    stride_mat2_k,
    stride_mat2_n,
    stride_scales_m,
    stride_output_m,
    stride_output_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_LARGE_BLOCK: tl.constexpr,
):
    # Grid: (M / BLOCK_SIZE_M, N / BLOCK_SIZE_N)
    if USE_LARGE_BLOCK:
        pid_m = tle.program_id(0)
        pid_n = tle.program_id(1)
    else:
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

    # Offsets for output positions
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Initialize accumulator in float32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Offsets for K dimension
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Load scales for this row block
    scales_ptrs = scales_ptr + offs_m * stride_scales_m
    scales = tl.load(scales_ptrs, mask=offs_m < M, other=1.0)

    # Iterate over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Compute offsets for weight (packed: K//2 columns)
        # Each packed byte contains 2 4-bit values
        # We need to load the packed bytes and extract nibbles

        # weight_offs_k goes from k*BLOCK_SIZE_K to (k+1)*BLOCK_SIZE_K - 1
        weight_offs_k = k * BLOCK_SIZE_K + offs_k

        # For each K position j, we need to read from column j//2
        # This gives us the same byte for pairs of j values
        packed_col_idx = weight_offs_k // 2

        # Load packed weight - same column for both low and high nibble
        weight_packed = tl.load(
            weight_ptr
            + offs_m[:, None] * stride_weight_m
            + packed_col_idx[None, :] * stride_weight_k,
            mask=(offs_m[:, None] < M) & (packed_col_idx[None, :] < K // 2),
            other=0.0,
        ).to(tl.uint8)

        # Extract low nibble (bits 0-3) - for even K indices
        # Extract high nibble (bits 4-7) - for odd K indices
        w_even = (weight_packed & 0x0F).to(tl.float32)  # even indices
        w_odd = ((weight_packed >> 4) & 0x0F).to(tl.float32)  # odd indices

        # Select based on whether K index is even or odd
        # If offs_k % 2 == 0 (even), use w_even; else use w_odd
        w_combined = tl.where(offs_k[None, :] % 2 == 0, w_even, w_odd)

        # Load mat2 (activation) - shape (K, N)
        # Convert to float32 for the dot product
        mat2_ptrs = (
            mat2_ptr
            + weight_offs_k[:, None] * stride_mat2_k
            + offs_n[None, :] * stride_mat2_n
        )
        mat2 = tl.load(
            mat2_ptrs,
            mask=(weight_offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float32)

        # Compute partial matmul (both operands are now float32)
        accumulator += tl.dot(w_combined, mat2, allow_tf32=False)

    # Apply scales - scales is (M,), need to broadcast to (BLOCK_SIZE_M, BLOCK_SIZE_N)
    # Each row has its own scale
    scales_broadcast = scales[:, None]  # (BLOCK_SIZE_M, 1)
    accumulator = accumulator * scales_broadcast

    # Store result as float32
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = (
        output_ptr
        + offs_cm[:, None] * stride_output_m
        + offs_cn[None, :] * stride_output_n
    )
    output_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=output_mask)


def _weight_int8pack_mm(weight, mat2, scales):
    """
    Perform quantized weight-only matrix multiplication.

    Args:
        weight: Packed int8 weight tensor of shape (M, K//2)
        mat2: Activation tensor of shape (K, N)
        scales: Scale tensor of shape (M,)

    Returns:
        Output tensor of shape (M, N)
    """
    logger.debug("METAX GEMS WEIGHT_INT8PACK_MM")

    # Get dimensions
    M = weight.shape[0]
    K = weight.shape[1] * 2  # Packed dimension
    N = mat2.shape[1]

    logger.debug(
        "METAX GEMS WEIGHT_INT8PACK_MM, [shape info]: M=%s, N=%s, K=%s",
        M,
        N,
        K,
    )

    # Allocate output as float32
    output = torch.empty((M, N), device=weight.device, dtype=torch.float32)

    # Define block sizes - need at least 16 for tl.dot
    # Use tile sizes that work with tl.dot requirements
    if K >= 64:
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 16
        BLOCK_SIZE_K = 64
    elif K >= 32:
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 16
        BLOCK_SIZE_K = 32
    else:
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 16
        BLOCK_SIZE_K = 16

    # Grid
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    with torch_device_fn.device(weight.device):
        weight_int8pack_mm_kernel[grid](
            weight,
            mat2,
            scales,
            output,
            M,
            N,
            K,
            weight.stride(0),
            weight.stride(1),
            mat2.stride(0),
            mat2.stride(1),
            scales.stride(0),
            output.stride(0),
            output.stride(1),
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
        )

    return output