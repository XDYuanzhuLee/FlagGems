import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)

# Use a fixed small block size to avoid private memory issues on Metax
BLOCK_SIZE = 1024


@libentry()
@triton.jit(do_not_specialize=["eps"])
def add_layernorm_kernel(
    x_ptr,  # pointer to input
    residual_ptr,  # pointer to residual
    weight_ptr,  # pointer to weight
    bias_ptr,  # pointer to bias
    out_ptr,  # pointer to output
    M,  # number of rows
    N,  # number of columns
    eps,  # epsilon
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    row_offset = pid * N

    # Compute mean and variance using a loop
    sum_val = tl.zeros((1,), dtype=tl.float32)
    sum_sq = tl.zeros((1,), dtype=tl.float32)
    cnt = 0

    for start_n in range(0, N, BLOCK_N):
        n_offsets = start_n + tl.arange(0, BLOCK_N)
        mask = n_offsets < N
        x = tl.load(x_ptr + row_offset + n_offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + row_offset + n_offsets, mask=mask, other=0.0).to(tl.float32)
        x = x + residual
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        cnt += tl.sum(mask.to(tl.int32), axis=0)

    mean = sum_val / N
    var = sum_sq / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize and store
    for start_n in range(0, N, BLOCK_N):
        n_offsets = start_n + tl.arange(0, BLOCK_N)
        mask = n_offsets < N
        x = tl.load(x_ptr + row_offset + n_offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + row_offset + n_offsets, mask=mask, other=0.0).to(tl.float32)
        x = x + residual

        # Load weight and bias
        if weight_ptr is None:
            w = 1.0
        else:
            w = tl.load(weight_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)

        if bias_ptr is None:
            b = 0.0
        else:
            b = tl.load(bias_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)

        # Compute output: (x - mean) * rstd * weight + bias
        out = (x - mean) * rstd * w + b
        out = out.to(out_ptr.dtype.element_ty)

        tl.store(out_ptr + row_offset + n_offsets, out, mask=mask)


def add_layernorm(input, residual, normalized_shape, weight=None, bias=None, eps=1e-5):
    logger.debug("METAX GEMS ADD LAYERNORM FORWARD")

    dim = input.ndim - len(normalized_shape)
    M = math.prod(input.shape[:dim])
    N = math.prod(normalized_shape)

    assert input.shape == residual.shape, "Input and residual must have the same shape"

    input = input.contiguous()
    residual = residual.contiguous()
    weight = None if weight is None else weight.contiguous()
    bias = None if bias is None else bias.contiguous()

    y = torch.empty_like(input)

    grid = (M, 1, 1)

    with torch_device_fn.device(input.device):
        add_layernorm_kernel[grid](
            input,
            residual,
            weight,
            bias,
            y,
            M,
            N,
            eps,
            BLOCK_N=BLOCK_SIZE,
        )

    return y