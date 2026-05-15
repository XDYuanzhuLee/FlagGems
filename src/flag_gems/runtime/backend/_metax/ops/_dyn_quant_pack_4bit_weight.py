import logging

import torch

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)


def _dyn_quant_pack_4bit_weight(
    weights: torch.Tensor,
    scales_zeros: torch.Tensor,
    bias: torch.Tensor,
    block_size: int,
    in_features: int,
    out_features: int,
):
    """
    Pack 4-bit quantized weights for efficient storage and computation.

    Args:
        weights: Input weight tensor of shape [out_features, in_features]
        scales_zeros: Quantization scales and zeros of shape [out_features, num_groups * 2]
        bias: Optional bias of shape [out_features]
        block_size: Block size for quantization
        in_features: Input feature dimension
        out_features: Output feature dimension

    Returns:
        Packed 4-bit weights tensor
    """
    logger.debug("METAX GEMS _DYN_QUANT_PACK_4BIT_WEIGHT")

    # Validate input shapes
    assert weights.shape == (out_features, in_features), f"Invalid weights shape: {weights.shape}, expected ({out_features}, {in_features})"

    num_groups = (in_features + block_size - 1) // block_size
    expected_scales_zeros_shape = (out_features, num_groups * 2)
    if scales_zeros.shape != expected_scales_zeros_shape:
        logger.warning(f"scales_zeros shape {scales_zeros.shape} doesn't match expected {expected_scales_zeros_shape}, adjusting num_groups")
        num_groups = scales_zeros.shape[1] // 2

    # Compute output shape: pack 2 4-bit values into 1 byte
    n_packed_elements = (in_features + 1) // 2  # Round up to pack odd number of values
    output_shape = (out_features, n_packed_elements)
    output = torch.empty(output_shape, dtype=torch.uint8, device=weights.device)

    # Quantize weights to 4-bit using scales
    # For each block, we quantize using the corresponding scale and zero point
    quantized_weights = torch.empty_like(weights, dtype=torch.uint8)

    # Simple per-element quantization: scale * round(weight / scale)
    # This is a simplified version - real implementation would use block-wise quantization
    for out_feat in range(out_features):
        for block_idx in range(num_groups):
            block_start = block_idx * block_size
            block_end = min(block_start + block_size, in_features)
            # scales_zeros is [out_features, num_groups * 2], interleaved [scale, zero]
            scale = scales_zeros[out_feat, block_idx * 2].abs() + 1e-6  # Avoid division by zero
            zero = scales_zeros[out_feat, block_idx * 2 + 1] if block_idx * 2 + 1 < scales_zeros.shape[1] else 0.0

            block_weights = weights[out_feat, block_start:block_end]
            quantized_block = ((block_weights - zero) / scale).round().clamp(0, 15).to(torch.uint8)
            quantized_weights[out_feat, block_start:block_end] = quantized_block

    # Pack 2 4-bit values into 1 byte using bitwise operations
    # Create views for even and odd indices
    even_indices = quantized_weights[:, 0::2]  # Lower nibble
    odd_indices = quantized_weights[:, 1::2]   # Upper nibble

    # Pad odd_indices if needed for odd in_features
    if in_features % 2 == 1:
        odd_indices_padded = torch.zeros(out_features, n_packed_elements, dtype=torch.uint8, device=weights.device)
        odd_indices_padded[:, :odd_indices.shape[1]] = odd_indices
        odd_indices = odd_indices_padded

    # Pack: lower_nibble | (upper_nibble << 4)
    output = (even_indices | (odd_indices << 4)).to(torch.uint8)

    return output


def dyn_quant_pack_4bit_weight(
    weights: torch.Tensor,
    scales_zeros: torch.Tensor,
    bias: torch.Tensor,
    block_size: int,
    in_features: int,
    out_features: int,
):
    """
    Wrapper function for _dyn_quant_pack_4bit_weight with logging.
    """
    return _dyn_quant_pack_4bit_weight(weights, scales_zeros, bias, block_size, in_features, out_features)