# Simple test file for _dyn_quant_pack_4bit_weight operator
import pytest
import torch

import flag_gems

device = flag_gems.device


def _reference_dyn_quant_pack_4bit_weight(
    weights: torch.Tensor,
    scales_zeros: torch.Tensor,
    block_size: int,
    in_features: int,
    out_features: int,
):
    """
    Reference implementation for 4-bit weight quantization packing.
    """
    num_groups = (in_features + block_size - 1) // block_size

    # Quantize weights to 4-bit using scales
    quantized_weights = torch.empty_like(weights, dtype=torch.uint8)

    for out_feat in range(out_features):
        for block_idx in range(num_groups):
            block_start = block_idx * block_size
            block_end = min(block_start + block_size, in_features)
            # scales_zeros is [out_features, num_groups * 2], interleaved [scale, zero]
            scale = scales_zeros[out_feat, block_idx * 2].abs() + 1e-6
            zero = scales_zeros[out_feat, block_idx * 2 + 1] if block_idx * 2 + 1 < scales_zeros.shape[1] else 0.0

            block_weights = weights[out_feat, block_start:block_end]
            quantized_block = ((block_weights - zero) / scale).round().clamp(0, 15).to(torch.uint8)
            quantized_weights[out_feat, block_start:block_end] = quantized_block

    # Pack 2 4-bit values into 1 byte
    n_packed = (in_features + 1) // 2
    output = torch.empty((out_features, n_packed), dtype=torch.uint8, device=weights.device)

    for out_feat in range(out_features):
        for i in range(n_packed):
            lower = quantized_weights[out_feat, 2 * i].item() if 2 * i < in_features else 0
            upper = quantized_weights[out_feat, 2 * i + 1].item() if 2 * i + 1 < in_features else 0
            output[out_feat, i] = lower | (upper << 4)

    return output


@pytest.mark.dyn_quant_pack_4bit_weight
@pytest.mark.parametrize("out_features", [32, 64])
@pytest.mark.parametrize("in_features", [128, 256])
@pytest.mark.parametrize("block_size", [32, 64])
def test_accuracy_dyn_quant_pack_4bit_weight(out_features, in_features, block_size):
    """Test accuracy of _dyn_quant_pack_4bit_weight operator"""
    dtype = torch.float16

    # Adjust block_size if it's larger than in_features
    block_size = min(block_size, in_features)

    weights = torch.randn(out_features, in_features, dtype=dtype, device=device)

    num_groups = (in_features + block_size - 1) // block_size
    scales_zeros = torch.randn(out_features, num_groups * 2, dtype=dtype, device=device)

    # Run reference implementation
    ref_out = _reference_dyn_quant_pack_4bit_weight(
        weights, scales_zeros, block_size, in_features, out_features
    )

    # Run gems implementation - call via flag_gems namespace directly
    res_out = flag_gems.dyn_quant_pack_4bit_weight(weights, scales_zeros, None, block_size, in_features, out_features)

    # Compare
    torch.testing.assert_close(res_out, ref_out)