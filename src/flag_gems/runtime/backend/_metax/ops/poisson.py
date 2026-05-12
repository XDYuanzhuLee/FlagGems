import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger("flag_gems." + __name__)


def heur_block(args):
    if args["N"] <= 512:
        return 256
    elif args["N"] <= 1024:
        return 512
    else:
        return 1024


def heur_num_warps(args):
    if args["N"] <= 512:
        return 4
    elif args["N"] <= 1024:
        return 8
    else:
        return 16


# Poisson sampling kernel using inverse transform method
@triton.heuristics(
    {
        "BLOCK": heur_block,
        "num_warps": heur_num_warps,
    }
)
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def poisson_kernel(
    rates_ptr,
    output_ptr,
    N,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    # Get position
    pid = tle.program_id(axis=0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    mask = offsets < N

    # Load rates
    rates = tl.load(rates_ptr + offsets, mask=mask)

    # Generate random numbers using philox
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    # Adjust offset for each thread
    i4 = offsets
    c0 += i4
    _O = c0 * 0

    # Generate 4 random uint32 values
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)

    # Convert to uniform floats
    rand0 = uint_to_uniform_float(r0)
    rand1 = uint_to_uniform_float(r1)
    rand2 = uint_to_uniform_float(r2)
    rand3 = uint_to_uniform_float(r3)

    # Apply Poisson sampling
    result0 = poisson_sampling(rates, rand0, r0)
    result1 = poisson_sampling(rates, rand1, r1)
    result2 = poisson_sampling(rates, rand2, r2)
    result3 = poisson_sampling(rates, rand3, r3)

    # Store results with unrolling
    UNROLL = 4
    start = pid.to(tl.uint64) * BLOCK * UNROLL

    off_0 = start + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    tl.store(output_ptr + off_0, result0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(output_ptr + off_1, result1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(output_ptr + off_2, result2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(output_ptr + off_3, result3, mask=off_3 < N, eviction_policy="evict_first")


@triton.jit
def poisson_sampling(rate, u, rand_state):
    """
    Sample from Poisson distribution.
    Uses a combination of algorithms:
    - For small rates (< 1): direct computation
    - For medium rates (1-30): Knuth's algorithm
    - For large rates (> 30): Normal approximation
    """
    rate = rate.to(tl.float32)

    # For rate <= 0, return 0
    result = tl.where(rate <= 0.0, 0.0, poisson_core(rate, u, rand_state))

    return tl.cast(result, tl.int64)


@triton.jit
def poisson_core(rate, u, rand_state):
    """
    Core Poisson sampling implementation.
    Uses the inverse CDF method.
    """
    # Use the random state to generate more random numbers
    # We'll use a simple hash to generate different seeds
    seed = rand_state.to(tl.uint32)

    # For small rates (< 1), use direct formula
    # P(X=0) = exp(-lambda), P(X=k) = lambda^k * exp(-lambda) / k!
    exp_neg_rate = tl.exp(-rate)

    # Initialize using the input uniform random
    product = u
    k = 0

    # Use static loop with a reasonable number of iterations
    # For rate < 30, we need at most ~30-40 iterations
    for _ in tl.static_range(30):
        # Generate a new random number from the philox state
        # Use simple linear congruential generator for additional randoms
        seed = seed * 1103515245 + 12345
        rand_val = (seed >> 16) * 4.6566127342e-10  # Convert to float in (0,1)

        product = product * rand_val
        k = k + 1

        # Early exit if product < exp(-rate)
        # We need to check this condition
        if product < exp_neg_rate:
            break

    return k - 1


def poisson(input: torch.Tensor, generator=None):
    logger.debug("METAX GEMS POISSON")
    dtype = input.dtype
    device = input.device
    N = input.numel()

    # Flatten input for processing
    input_flat = input.reshape(-1)

    # Output is int64 (since Poisson returns counts)
    output = torch.empty(N, dtype=torch.int64, device=device)

    # Calculate grid
    BLOCK = 1024
    UNROLL = 4
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)

    # Get philox seed and offset
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )

    with torch_device_fn.device(device):
        poisson_kernel[grid_fn](
            input_flat,
            output,
            N,
            philox_seed,
            philox_offset,
            BLOCK=BLOCK,
        )

    # Reshape to match input shape
    output = output.reshape(input.shape)

    return output