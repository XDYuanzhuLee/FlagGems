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


# Simple Poisson sampling kernel using normal approximation
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

    # Generate random values
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)

    # Convert to uniform floats
    u0 = uint_to_uniform_float(r0)
    u1 = uint_to_uniform_float(r1)
    u2 = uint_to_uniform_float(r2)
    u3 = uint_to_uniform_float(r3)

    # Apply Poisson sampling using normal approximation
    result0 = poisson_sample(rates, u0)
    result1 = poisson_sample(rates, u1)
    result2 = poisson_sample(rates, u2)
    result3 = poisson_sample(rates, u3)

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
def poisson_sample(rate, u):
    """
    Sample from Poisson distribution using simplified normal approximation.
    """
    rate = rate.to(tl.float32)

    # For rate <= 0, return 0
    result = tl.where(rate <= 0.0, 0.0, poisson_normal_approx(rate, u))

    return tl.cast(result, tl.int64)


@triton.jit
def poisson_normal_approx(rate, u):
    """
    Simple Poisson sampling using normal approximation.
    Poisson(lambda) ~ Normal(lambda, sqrt(lambda))
    """
    # Use simple inverse CDF approximation
    # For efficiency, use a simpler formula

    # Transform u to normal using simple approximation
    # Use the probit function approximation

    # Simple polynomial approximation for normal inverse CDF
    y = u - 0.5

    # Very simple approximation
    if y < 0:
        # Use a simple formula for negative side
        x = -tl.sqrt(-2.0 * tl.log(u + 1e-10))
    else:
        # Use a simple formula for positive side
        x = tl.sqrt(-2.0 * tl.log(1.0 - u + 1e-10))

    # Transform to Poisson
    mu = rate
    sigma = tl.sqrt(rate)

    # Add continuity correction
    result = mu + sigma * x + 0.5

    # Floor and ensure non-negative
    result = tl.floor(result)
    result = tl.where(result < 0, 0.0, result)

    return result


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