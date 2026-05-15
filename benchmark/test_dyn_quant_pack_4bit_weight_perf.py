import pytest
import torch
import time

import flag_gems


def time_function(fn, *args, warmup=10, repeat=100, **kwargs):
    """Simple timing utility"""
    # Warmup
    for _ in range(warmup):
        _ = fn(*args, **kwargs)
    torch.cuda.synchronize()

    # Timing
    start = time.perf_counter()
    for _ in range(repeat):
        _ = fn(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / repeat * 1000  # ms


def reference_impl(weights, scales_zeros, block_size, in_features, out_features):
    """Simple reference implementation for timing comparison"""
    num_groups = (in_features + block_size - 1) // block_size
    quantized_weights = torch.empty_like(weights, dtype=torch.uint8)

    for out_feat in range(out_features):
        for block_idx in range(num_groups):
            block_start = block_idx * block_size
            block_end = min(block_start + block_size, in_features)
            scale = scales_zeros[out_feat, block_idx * 2].abs() + 1e-6
            zero = scales_zeros[out_feat, block_idx * 2 + 1] if block_idx * 2 + 1 < scales_zeros.shape[1] else 0.0
            block_weights = weights[out_feat, block_start:block_end]
            quantized_block = ((block_weights - zero) / scale).round().clamp(0, 15).to(torch.uint8)
            quantized_weights[out_feat, block_start:block_end] = quantized_block

    n_packed_elements = (in_features + 1) // 2
    even_indices = quantized_weights[:, 0::2]
    odd_indices = quantized_weights[:, 1::2]

    if in_features % 2 == 1:
        odd_indices_padded = torch.zeros(out_features, n_packed_elements, dtype=torch.uint8, device=weights.device)
        odd_indices_padded[:, :odd_indices.shape[1]] = odd_indices
        odd_indices = odd_indices_padded

    output = (even_indices | (odd_indices << 4)).to(torch.uint8)
    return output


@pytest.mark.dyn_quant_pack_4bit_weight
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [
    (1024, 4096),
    (2048, 2048),
    (256, 512),
])
def test_perf_dyn_quant_pack_4bit_weight(dtype, shape):
    """Benchmark for _dyn_quant_pack_4bit_weight operator"""
    out_features, in_features = shape
    block_size = 128

    num_groups = (in_features + block_size - 1) // block_size
    weights = torch.randn(out_features, in_features, device=flag_gems.device, dtype=dtype)
    scales_zeros = torch.randn(out_features, num_groups * 2, device=flag_gems.device, dtype=dtype)

    # Time the gems implementation
    gems_latency = time_function(
        flag_gems.dyn_quant_pack_4bit_weight,
        weights, scales_zeros, None, block_size, in_features, out_features,
        warmup=3, repeat=20
    )

    # Time the reference implementation (pure PyTorch loop-based)
    ref_latency = time_function(
        reference_impl,
        weights, scales_zeros, block_size, in_features, out_features,
        warmup=3, repeat=20
    )

    # Calculate speedup
    speedup = ref_latency / gems_latency if gems_latency > 0 else 0

    # Print results
    print(f"\nOperator: _dyn_quant_pack_4bit_weight")
    print(f"Performance Test (dtype={dtype}, shape={shape})")
    print(f"SUCCESS    {ref_latency:.4f}    {gems_latency:.4f}    {speedup:.3f}")

    # The speedup should be > 0.5 for acceptable performance
    assert speedup >= 0.5, f"Performance is too slow: speedup={speedup}"