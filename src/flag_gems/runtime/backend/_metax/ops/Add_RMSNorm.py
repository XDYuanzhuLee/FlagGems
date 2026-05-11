import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def add_rms_norm_kernel(
    output_ptr,  # pointer to the output
    input_ptr,  # pointer to the input x
    residual_ptr,  # pointer to the residual
    w_ptr,  # pointer to the weights
    in_stride_r,  # how much to increase the pointer when moving by 1 row
    in_stride_c,  # how much to increase the pointer when moving by 1 col
    r_stride_r,  # how much to increase the pointer when moving by 1 row
    r_stride_c,  # how much to increase the pointer when moving by 1 col
    out_stride_r,  # output row stride
    out_stride_c,  # output col stride
    N,  # number of columns in in_ptr
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(input_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        input_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = input_ptr.dtype.element_ty

    pid = tle.program_id(0)
    input_ptr += pid * in_stride_r
    residual_ptr += pid * r_stride_r
    output_ptr += pid * out_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)

    # Load input x
    x = tl.load(input_ptr + cols * in_stride_c, mask, other=0.0).to(cdtype)
    # Load residual
    r = tl.load(residual_ptr + cols * r_stride_c, mask, other=0.0).to(cdtype)

    # Compute x + residual
    x = x + r

    # Compute RMSNorm
    var = tl.sum(x * x / N, axis=0)
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms * w).to(cdtype)

    # Store output
    tl.store(output_ptr + cols * out_stride_c, y, mask=mask)


def add_rms_norm(x, residual, normalized_shape, weight, eps=1e-5):
    """
    Computes Add_RMSNorm: RMSNorm(x + residual)

    This function performs element-wise addition of input and residual,
    then applies RMS normalization.

    Args:
        x: Input tensor
        residual: Residual tensor to add
        normalized_shape: Shape to normalize over
        weight: RMSNorm weight
        eps: Epsilon for numerical stability

    Returns:
        Normalized output tensor
    """
    logger.debug(
        "METAX GEMS ADD_RMS_NORM FORWARD, [input shape]: %s, [residual shape]: %s, [weight shape]: %s",
        x.size(),
        residual.size(),
        weight.size(),
    )

    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    BLOCK_SIZE = triton.next_power_of_2(N)
    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()
    output = torch.empty_like(x)

    with torch_device_fn.device(x.device):
        add_rms_norm_kernel[M,](
            output,
            x,
            residual,
            weight,
            N,  # in_stride_r - row stride (distance between rows)
            1,  # in_stride_c - column stride (contiguous)
            N,  # r_stride_r
            1,  # r_stride_c
            N,  # out_stride_r
            1,  # out_stride_c
            N,
            eps,
            BLOCK_SIZE,
        )
    return output