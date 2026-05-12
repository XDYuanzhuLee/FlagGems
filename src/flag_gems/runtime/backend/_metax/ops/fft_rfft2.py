import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)

# Get math functions from tl_extra_shim for cross-backend compatibility
sin = tl_extra_shim.sin
cos = tl_extra_shim.cos
exp = tl_extra_shim.exp
exp2 = tl_extra_shim.exp2
rsqrt = tl_extra_shim.rsqrt


@libentry()
@triton.jit
def rfft2_kernel(
    input_ptr,
    output_real_ptr,
    output_imag_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    n_fft_cols: tl.constexpr,
):
    """Compute 1D FFT along the last dimension for each row.

    Uses naive DFT for simplicity - suitable for small FFT sizes.
    For production use, radix-2 or split-radix FFT should be used.
    """
    pid = tle.program_id(0)
    row = pid

    if row >= n_rows:
        return

    # Compute RFFT for each output frequency bin
    # For real input, output is n_fft_cols = n_fft // 2 + 1
    for freq_idx in range(n_fft_cols):
        # Compute sum_{k=0}^{n_fft-1} x[k] * exp(-2*pi*i*k*freq/n_fft)
        sum_real = 0.0
        sum_imag = 0.0

        for k in range(n_cols):
            # Load input element
            x_k = tl.load(input_ptr + row * n_cols + k).to(tl.float32)

            # Twiddle factor: W = exp(-2*pi*i*k*freq/n_fft)
            # = cos(2*pi*k*freq/n_fft) - i*sin(2*pi*k*freq/n_fft)
            angle = -2.0 * 3.141592653589793 * k * freq_idx / n_fft_cols

            w_real = cos(angle)
            w_imag = sin(angle)

            # Accumulate: x_k * W = x_k * (w_real - i*w_imag)
            # = x_k * w_real - i * x_k * w_imag
            sum_real = sum_real + x_k * w_real
            sum_imag = sum_imag + x_k * w_imag

        # Store result
        out_idx = row * n_fft_cols + freq_idx
        tl.store(output_real_ptr + out_idx, sum_real)
        tl.store(output_imag_ptr + out_idx, sum_imag)


def fft_rfft2(input: torch.Tensor, s=None, dim=(-2, -1), norm=None):
    """2D real FFT for Metax backend.

    Computes the 2-dimensional discrete Fourier transform of real input.
    This is a specialized implementation for the Metax backend.

    Args:
        input: Input tensor (real-valued)
        s: Output size tuple (optional)
        dim: Dimensions to transform (default: (-2, -1))
        norm: Normalization mode (None, "forward", "backward", "ortho")

    Returns:
        Complex tensor of shape (..., n_rows, n_fft_cols) where
        n_fft_cols = n_cols // 2 + 1 for real input
    """
    logger.debug("METAX GEMS FFT_RFFT2")

    # Validate input
    if dim != (-2, -1):
        raise ValueError("Only dim=(-2, -1) is supported for fft_rfft2")

    # Handle the case where input has more than 2 dimensions
    if input.dim() > 2:
        # Batch over leading dimensions
        batch_size = 1
        for i in range(input.dim() - 2):
            batch_size *= input.shape[i]
        input = input.reshape(batch_size, input.shape[-2], input.shape[-1])

    n_rows = input.shape[-2]
    n_cols = input.shape[-1]

    # Handle s parameter for output size
    if s is None:
        n_fft_cols = n_cols // 2 + 1  # For real input
    else:
        n_fft_cols = s[1] if len(s) > 1 else n_cols // 2 + 1

    # Pad input if necessary (for s parameter)
    if s is not None and len(s) > 1 and s[1] > n_cols:
        pad_size = s[1] - n_cols
        input = torch.nn.functional.pad(input, (0, pad_size), mode='constant', value=0)
        n_cols = input.shape[-1]

    # Allocate output tensors (real and imaginary parts)
    # Use float32 to match PyTorch's default FFT precision for float32 input
    output_dtype = torch.float32

    # Allocate separate real and imaginary tensors
    output_real = torch.empty((*input.shape[:-2], n_rows, n_fft_cols),
                              dtype=output_dtype, device=input.device)
    output_imag = torch.empty((*input.shape[:-2], n_rows, n_fft_cols),
                              dtype=output_dtype, device=input.device)

    # Ensure input is contiguous
    input = input.contiguous()

    # Launch Triton kernel
    grid = (n_rows,)

    with torch_device_fn.device(input.device):
        rfft2_kernel[grid](
            input,
            output_real,
            output_imag,
            n_rows,
            n_cols,
            n_fft_cols,
        )

    # Combine real and imaginary parts into complex tensor
    result = torch.complex(output_real, output_imag)

    # Apply normalization
    if norm == "forward":
        result = result / (n_rows * n_cols)
    elif norm == "ortho":
        result = result / (n_rows * n_cols) ** 0.5

    return result