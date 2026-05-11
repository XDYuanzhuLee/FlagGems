import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)


@libentry()
@triton.jit
def reduce_norm_kernel_1(X, Mid, M, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0).to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    X = X + offset
    Mid = Mid + pid
    mask = offset < M

    x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
    mid = tl.sum(x * x)
    tl.store(Mid, mid)


@libentry()
@triton.jit
def reduce_norm_kernel_2(Mid, Out, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    Mid = Mid + offset
    mask = offset < MID_SIZE
    mid = tl.load(Mid, mask=mask, other=0.0).to(tl.float32)
    out = tl.sqrt(tl.sum(mid))
    tl.store(Out, out)


def reduce_norm(inp, dim=None, keepdim=False):
    """
    Compute the L2 norm (Euclidean norm) of the input tensor.

    This is the Metax-specific implementation.

    Args:
        inp: Input tensor
        dim: Dimension(s) along which to compute the norm. If None, computes norm over all elements.
        keepdim: If True, the output tensor has dim retained as dimension of size 1.

    Returns:
        The L2 norm of the input tensor.
    """
    logger.debug("METAX GEMS REDUCE NORM")
    dtype = inp.dtype
    if dtype not in [torch.float16, torch.float32, torch.bfloat16]:
        raise NotImplementedError(f"reduce_norm not implemented for {dtype}")

    # Track if dim was originally None for later handling
    dim_was_none = dim is None
    orig_dim = dim

    with torch_device_fn.device(inp.device):
        # Handle scalar output case (dim=None or dim covers all dimensions)
        if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == inp.ndim):
            # Compute norm over all elements (flatten)
            dim = list(range(inp.ndim))
            shape = [1] * inp.ndim
            inp = dim_compress(inp, dim)
            M = inp.numel()
            BLOCK_SIZE = triton.next_power_of_2(math.ceil(math.sqrt(M)))
            MID_SIZE = triton.cdiv(M, BLOCK_SIZE)
            BLOCK_MID = triton.next_power_of_2(MID_SIZE)

            mid = torch.empty([MID_SIZE], dtype=torch.float32, device=inp.device)
            out = torch.empty(shape, dtype=dtype, device=inp.device)

            reduce_norm_kernel_1[(MID_SIZE,)](inp, mid, M, BLOCK_SIZE)
            reduce_norm_kernel_2[(1,)](mid, out, MID_SIZE, BLOCK_MID)
        else:
            # For dim-specific case, use a simple approach:
            # Compute sum of squares along dim, then sqrt
            shape = list(inp.shape)
            if isinstance(dim, int):
                dim = [dim]
            dim = [d % inp.ndim for d in dim]

            # Compute the output shape
            output_shape = list(inp.shape)
            for d in dim:
                output_shape[d] = 1

            # Flatten input along reduction dimensions
            inp_compressed = dim_compress(inp, dim)
            N = 1
            for d in dim:
                N *= shape[d]
            M = inp_compressed.numel() // N

            # For each "row" in the non-reduction dimensions, compute the norm
            out = torch.empty(output_shape, dtype=dtype, device=inp.device)

            BLOCK_SIZE = triton.next_power_of_2(math.ceil(math.sqrt(N)))
            MID_SIZE = triton.cdiv(N, BLOCK_SIZE)
            BLOCK_MID = triton.next_power_of_2(MID_SIZE)

            # Process each row
            for m_idx in range(M):
                row_start = m_idx * N
                row_data = inp_compressed[row_start:row_start + N].contiguous()

                mid = torch.empty([MID_SIZE], dtype=torch.float32, device=inp.device)
                row_out = torch.empty([1], dtype=dtype, device=inp.device)

                reduce_norm_kernel_1[(MID_SIZE,)](row_data, mid, N, BLOCK_SIZE)
                reduce_norm_kernel_2[(1,)](mid, row_out, MID_SIZE, BLOCK_MID)

                # Store result at the correct position
                out_view = out
                for d in range(out.ndim):
                    if d == dim[0]:
                        out_view = out_view[m_idx]
                    else:
                        out_view = out_view[0]
                out_view.copy_(row_out)

            # Handle keepdim
            if not keepdim:
                for d in sorted(dim, reverse=True):
                    out = out.squeeze(dim=d)

            return out

    # Handle the case when dim was originally None
    if dim_was_none and not keepdim:
        out = out.squeeze()

    return out