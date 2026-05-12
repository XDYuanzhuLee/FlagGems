import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)


@libentry()
@triton.jit
def quantize_kernel(
    input_ptr,
    output_ptr,
    scale_value,
    zero_point_value,
    numel,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tle.program_id(0)

    # Calculate offset for this program
    block_start = pid * BLOCK_SIZE

    # Create offsets - use 1D linear offset
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create mask to guard against out-of-bounds
    mask = offsets < numel

    # Load input data - use linear 1D offset
    input = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Quantize formula: round(input / scale) + zero_point
    # Use: floor(x + 0.5) with small epsilon to handle floating point precision
    # at exact .5 boundaries
    scaled = input / scale_value
    # Use a small epsilon to avoid rounding up for values that are exactly x.5
    # This helps match PyTorch's behavior more closely
    epsilon = 1e-7
    rounded = tl.floor(scaled + 0.5 - epsilon)

    # Add zero_point
    quantized = rounded + zero_point_value

    # Clip to valid range [0, 255] for quint8 or [-128, 127] for qint8
    if dtype == 0:  # torch.qint8
        quantized = tl.minimum(quantized, tl.constexpr(127.0))
        quantized = tl.maximum(quantized, tl.constexpr(-128.0))
    else:  # torch.quint8
        quantized = tl.minimum(quantized, tl.constexpr(255.0))
        quantized = tl.maximum(quantized, tl.constexpr(0.0))

    # Cast to int8 or uint8 based on dtype
    if dtype == 0:  # torch.qint8
        quantized = quantized.to(tl.int8)
    else:  # torch.quint8
        quantized = quantized.to(tl.uint8)

    # Store output - use linear 1D offset
    tl.store(output_ptr + offsets, quantized, mask=mask)


def quantize(input: torch.Tensor, scale: float, zero_point: int, dtype: torch.dtype):
    r"""
    Quantizes a float tensor to a quantized tensor with given scale and zero point.

    Args:
        input: Input float tensor to quantize
        scale: Scale to apply in quantization formula
        zero_point: Offset in integer value that maps to float zero
        dtype: The desired data type of returned tensor (torch.quint8 or torch.qint8)

    Returns:
        A newly quantized tensor
    """
    logger.debug("METAX GEMS QUANTIZE")

    # Determine output dtype
    if dtype == torch.quint8:
        dtype_code = 1  # For kernel
        output_dtype = torch.uint8
    elif dtype == torch.qint8:
        dtype_code = 0  # For kernel
        output_dtype = torch.int8
    else:
        raise ValueError(f"Unsupported dtype: {dtype}. Only torch.quint8 and torch.qint8 are supported.")

    # Convert scale and zero_point to tensor if needed
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, dtype=torch.float32, device=input.device)
    if not isinstance(zero_point, torch.Tensor):
        zero_point = torch.tensor(zero_point, dtype=torch.int64, device=input.device)

    # Ensure input is contiguous for linear access
    input = input.contiguous()

    numel = input.numel()
    # Maintain original shape
    output = torch.empty(input.shape, dtype=output_dtype, device=input.device)

    # Define block size
    BLOCK_SIZE = 1024

    # Calculate grid
    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        quantize_kernel[grid](
            input,
            output,
            scale.item(),
            zero_point.item(),
            numel,
            dtype_code,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    # Create quantized tensor with scale and zero_point
    # Note: PyTorch's quantized tensor needs special handling
    result = torch._make_per_tensor_quantized_tensor(output, scale.item(), zero_point.item())

    return result


def dequantize(input: torch.Tensor):
    r"""
    Dequantizes a quantized tensor back to float.

    Args:
        input: Input quantized tensor

    Returns:
        A float tensor
    """
    logger.debug("METAX GEMS DEQUANTIZE")

    # Get scale and zero_point from quantized tensor
    # Convert to float for computation
    x = input.dequantize()

    return x