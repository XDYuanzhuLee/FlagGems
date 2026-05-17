import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)


def _weight_int4pack_mm_with_scales_and_zeros(weight, mat2, qGroupSize, qScale, qZeros):
    """
    Weight-only INT4 matrix multiplication with per-group scales and zero points.
    This is a fallback implementation that uses simple Python computation.
    For production, a proper Triton kernel should be implemented.
    """
    logger.debug("METAX GEMS _weight_int4pack_mm_with_scales_and_zeros")

    M, K = mat2.shape
    N = weight.shape[0]

    assert weight.shape == (N, K // 8)
    assert qScale.shape == (N // qGroupSize, K)
    assert qZeros.shape == (N // qGroupSize, K)

    logger.debug(
        "METAX GEMS _weight_int4pack_mm_with_scales_and_zeros, [shape info]: M=%s, N=%s, K=%s, qGroupSize=%s",
        M, N, K, qGroupSize
    )

    # Simple fallback: compute on CPU and copy back
    # For production, this should be replaced with a proper Triton kernel

    # Move all tensors to CPU
    weight_cpu = weight.cpu()
    mat2_cpu = mat2.cpu()
    qScale_cpu = qScale.cpu()
    qZeros_cpu = qZeros.cpu()

    # Create output buffer
    result_cpu = torch.zeros((M, N), dtype=torch.float32, device='cpu')

    # Unpack all int4 weights at once
    # weight shape: (N, K//8) - packed int4
    # We need to unpack to (N, K) int8
    weight_unpacked = torch.zeros((N, K), dtype=torch.int8, device='cpu')

    for k in range(K):
        byte_idx = k // 8
        if k % 8 < 4:
            # Lower 4 bits
            half_byte_idx = k % 4
            val = (weight_cpu[:, byte_idx] >> (half_byte_idx * 4)) & 0x0F
        else:
            # Upper 4 bits
            half_byte_idx = k % 4
            val = (weight_cpu[:, byte_idx] >> (half_byte_idx * 4 + 4)) & 0x0F

        # Sign extend: if >= 8, it's negative
        val = torch.where(val >= 8, val - 16, val)
        weight_unpacked[:, k] = val.to(torch.int8)

    # Apply scale and zero per group
    num_groups = N // qGroupSize
    weight_dequant = torch.zeros((N, K), dtype=torch.float32, device='cpu')

    for g in range(num_groups):
        start = g * qGroupSize
        end = (g + 1) * qGroupSize
        scale = qScale_cpu[g, :].to(torch.float32)
        zero = qZeros_cpu[g, :].to(torch.float32)
        weight_dequant[start:end, :] = (weight_unpacked[start:end, :].float() - zero) * scale

    # Matrix multiplication: mat2 @ weight_dequant^T
    mat2_float = mat2_cpu.to(torch.float32)
    result_cpu = mat2_float @ weight_dequant.t()

    # Convert to the same dtype as input mat2
    result = result_cpu.to(mat2.device).to(mat2.dtype)

    return result