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
def special_chebyshev_polynomial_v_kernel(
    inp_x,
    inp_n,
    out,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(inp_x + offsets, mask=mask, other=0.0)
    n = tl.load(inp_n + offsets, mask=mask, other=0)

    # Chebyshev polynomial of the third kind V_n(x)
    # V_n(x) = sin((n+1) * arccos(x)) / sin(arccos(x))
    # Using recurrence: V_0(x)=1, V_1(x)=2x-1, V_{n+1}(x)=2x*V_n(x)-V_{n-1}(x)
    # Compute using vectorized form

    # For small n, compute directly
    # For larger n, use the recurrence formula
    # Since n is typically small in practice, we handle common cases

    # Convert to float for computation
    x_f32 = x.to(tl.float32)
    n_i32 = n.to(tl.int32)

    # V_0(x) = 1
    v_prev2 = tl.full_like(x, 1.0, tl.float32)

    # V_1(x) = 2*x - 1
    v_prev1 = 2.0 * x_f32 - 1.0

    # Handle n=0 case
    result = tl.where(n_i32 == 0, v_prev2, tl.zeros_like(x, tl.float32))

    # Handle n=1 case
    result = tl.where(n_i32 == 1, v_prev1, result)

    # For n >= 2, use recurrence: V_{k+1} = 2*x*V_k - V_{k-1}
    # We need to iterate but Triton doesn't support dynamic loops well
    # So we unroll for small n or fall back to formula

    # For n >= 2, compute using the formula
    # Use vectorized computation for n >= 2
    # Since we can't do loops, compute up to a reasonable limit with unrolling

    # Fallback: compute using sin/cos formula for general n
    # arccos(x)
    arccos_x = tl.acos(tl.clamp(x_f32, -1.0, 1.0))
    # sin(arccos(x))
    sin_arccos = tl.sin(arccos_x)
    # sin((n+1) * arccos(x))
    n_f32 = n_i32.to(tl.float32)
    sin_term = tl.sin((n_f32 + 1.0) * arccos_x)
    # V_n(x) = sin((n+1) * arccos(x)) / sin(arccos(x))
    # Avoid division by zero
    result_n2 = tl.where(
        tl.abs(sin_arccos) > 1e-10,
        sin_term / sin_arccos,
        tl.zeros_like(x, tl.float32)
    )

    # Use result for n >= 2
    result = tl.where(n_i32 >= 2, result_n2, result)

    # Store result
    tl.store(out + offsets, result, mask=mask)


def special_chebyshev_polynomial_v(x, n):
    logger.debug("METAX GEMS SPECIAL_CHEBYSHEV_POLYNOMIAL_V")

    # Handle scalar n
    if not torch.is_tensor(n):
        n = torch.tensor(n, device=x.device, dtype=torch.int64)

    # Broadcast inputs
    output_shape = torch.broadcast_shapes(x.shape, n.shape)
    x = x.expand(output_shape)
    n = n.expand(output_shape)

    # Make contiguous
    x = x.contiguous()
    n = n.contiguous()

    output = torch.empty_like(x, dtype=torch.float32)

    N = output.numel()
    if N == 0:
        return output

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(x.device):
        special_chebyshev_polynomial_v_kernel[grid](
            x, n, output, N
        )

    return output.to(x.dtype)