import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger("flag_gems." + __name__)


@libentry()
@triton.jit
def abs_max_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    min_value = get_dtype_min(inp.type.element_ty)
    inp_val = tl.load(inp_ptrs, mask=mask, other=min_value)
    abs_val = tl.abs(inp_val)
    max_val = tl.max(abs_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, max_val)


@libentry()
@triton.jit
def abs_max_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    min_value = get_dtype_min(mid.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=min_value)
    max_val = tl.max(mid_val)
    tl.store(out, max_val)


def abs_max(inp):
    M = inp.numel()
    block_size = triton.next_power_of_2(max(1, triton.cdiv(M, 128)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(max(1, mid_size))

    dtype = inp.dtype
    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        abs_max_kernel_1[(mid_size, 1)](
            inp,
            mid,
            M,
            block_size,
        )
        abs_max_kernel_2[(1, 1)](
            mid, out, mid_size, block_mid
        )
    return out


@libentry()
@triton.jit
def dynamic_quantize_linear_kernel(
    x_ptr,
    scale_ptr,
    zero_point_ptr,
    output_ptr,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    scale = tl.load(scale_ptr)

    x_scaled = x * scale
    x_rounded = tl.round(x_scaled)
    x_clamped = tl.clamp(x_rounded, -128.0, 127.0)
    quantized = x_clamped.to(tl.int8)

    tl.store(output_ptr + offsets, quantized, mask=mask)


def dynamic_quantize_linear(x: torch.Tensor):
    logger.debug("METAX GEMS DYNAMICQUANTIZELINEAR")

    # Compute max(abs(x))
    max_abs = abs_max(x)

    # Compute scale: 127.0 / max_abs
    # Handle zero max_abs case
    scale = torch.where(
        max_abs == 0,
        torch.tensor(1.0, dtype=x.dtype, device=x.device),
        127.0 / max_abs
    )

    # Allocate output tensor
    output = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    zero_point = torch.tensor(0, dtype=torch.int8, device=x.device)

    # Launch kernel
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(x.device):
        dynamic_quantize_linear_kernel[grid](
            x,
            scale,
            zero_point,
            output,
            n_elements,
            BLOCK_SIZE=1024,
        )

    return output, scale.squeeze(), zero_point.squeeze()