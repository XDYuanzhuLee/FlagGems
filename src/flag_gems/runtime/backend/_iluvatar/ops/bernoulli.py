import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)

UNROLL = 4


configs = [
    triton.Config({"BLOCK": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK": 512}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 512}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=4),
]


@triton.autotune(configs=configs, key=["N"])
@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def bernoulli_kernel(
    probs,
    output,
    N,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    UNROLL: tl.constexpr = 4
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)

    off_0 = tl.program_id(0) * BLOCK * UNROLL + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    p0 = tl.load(probs + off_0, mask=off_0 < N, other=0.0, eviction_policy="evict_first")
    p1 = tl.load(probs + off_1, mask=off_1 < N, other=0.0, eviction_policy="evict_first")
    p2 = tl.load(probs + off_2, mask=off_2 < N, other=0.0, eviction_policy="evict_first")
    p3 = tl.load(probs + off_3, mask=off_3 < N, other=0.0, eviction_policy="evict_first")

    # Generate bernoulli random numbers: 1 if random < probability, else 0
    out0 = tl.where(r0 < p0, tl.constexpr(1.0), tl.constexpr(0.0))
    out1 = tl.where(r1 < p1, tl.constexpr(1.0), tl.constexpr(0.0))
    out2 = tl.where(r2 < p2, tl.constexpr(1.0), tl.constexpr(0.0))
    out3 = tl.where(r3 < p3, tl.constexpr(1.0), tl.constexpr(0.0))

    tl.store(output + off_0, out0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(output + off_1, out1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(output + off_2, out2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(output + off_3, out3, mask=off_3 < N, eviction_policy="evict_first")


def bernoulli(input, generator=None, *, out=None):
    logger.debug("ILUVATAR GEMS BERNOULLI")
    # Validate input probability range
    if torch.any(input < 0) or torch.any(input > 1):
        raise ValueError("概率值必须在 [0, 1] 范围内")

    # Use the same dtype as input for output if not specified
    if out is None:
        out = torch.empty_like(input, dtype=input.dtype)

    # Contiguous input for better performance
    input = input.contiguous()

    N = input.numel()
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)
    increment = triton.cdiv(N, UNROLL)

    philox_seed, philox_offset = philox_backend_seed_offset(increment)
    with torch_device_fn.device(input.device):
        bernoulli_kernel[grid_fn](
            input, out, N, philox_seed, philox_offset
        )

    return out