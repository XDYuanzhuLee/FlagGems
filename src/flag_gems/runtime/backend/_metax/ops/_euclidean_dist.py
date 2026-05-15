import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)


@triton.jit
def euclidean_dist_kernel(
    x1_ptr,
    x2_ptr,
    output_ptr,
    M,
    N,
    stride_x1_m,
    stride_x1_n,
    stride_x2_m,
    stride_x2_n,
    stride_out_m,
    stride_out_n,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid: M * M (each thread computes one element of the output)
    pid = tle.program_id(0)
    row = pid // M
    col = pid % M

    if row < M and col < M:
        # Load the two rows
        x1_row_ptr = x1_ptr + row * stride_x1_m
        x2_row_ptr = x2_ptr + col * stride_x2_m

        # Compute Euclidean distance using loop reduction
        sum_sq = 0.0
        for k in range(0, N, BLOCK_SIZE):
            k_offs = tl.arange(0, BLOCK_SIZE)
            mask = k_offs < N - k
            x1_vals = tl.load(x1_row_ptr + (k + k_offs) * stride_x1_n, mask=mask, other=0.0)
            x2_vals = tl.load(x2_row_ptr + (k + k_offs) * stride_x2_n, mask=mask, other=0.0)
            diff = x1_vals - x2_vals
            sum_sq += tl.sum(diff * diff, axis=0)

        dist = tl.sqrt(sum_sq)
        tl.store(output_ptr + row * stride_out_m + col, dist)


def euclidean_dist(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    logger.debug("METAX GEMS EUCLIDEAN_DIST")

    M = x1.shape[0]
    N = x1.shape[1]

    assert x1.shape == x2.shape, "Input shapes must match"
    assert x1.ndim == 2, "Inputs must be 2D"
    assert N > 0, "Feature dimension must be > 0"

    output = torch.empty((M, M), device=x1.device, dtype=x1.dtype)

    # Define block size for the reduction
    if N <= 32:
        BLOCK_SIZE = 32
    elif N <= 64:
        BLOCK_SIZE = 64
    elif N <= 128:
        BLOCK_SIZE = 128
    elif N <= 256:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = 512

    grid = lambda META: (M * M,)

    with torch_device_fn.device(x1.device):
        euclidean_dist_kernel[grid](
            x1,
            x2,
            output,
            M,
            N,
            x1.stride(0),
            x1.stride(1),
            x2.stride(0),
            x2.stride(1),
            output.stride(0),
            output.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return output