import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _permute_copy_kernel(
    x_ptr,
    y_ptr,
    numel,
    out_shape_ptr,
    in_strides_ptr,
    out_strides_ptr,
    perm_ptr,
    NDIMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    off = block_start + tl.arange(0, BLOCK_SIZE)
    mask = off < numel

    # Prepare offsets
    tmp = off.to(tl.int64)
    in_off = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    out_off = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    # Decompose linear index into multi-index over output shape
    # and accumulate input/output offsets using strides.
    # Iterate from last dim to first for divmod-based digit extraction.
    for rev_i in range(NDIMS):
        i = NDIMS - 1 - rev_i
        size_i = tl.load(out_shape_ptr + i)  # scalar broadcasted to vector
        # Avoid div by zero if size_i could be 0
        size_i = tl.where(size_i == 0, 1, size_i)
        idx_i = tmp % size_i
        tmp = tmp // size_i

        out_stride_i = tl.load(out_strides_ptr + i)
        perm_i = tl.load(perm_ptr + i)
        in_stride_axis = tl.load(in_strides_ptr + perm_i)

        out_off += idx_i * out_stride_i
        in_off += idx_i * in_stride_axis

    x = tl.load(x_ptr + in_off, mask=mask, other=0)
    tl.store(y_ptr + out_off, x, mask=mask)


def _normalize_dims(dims, ndim):
    if isinstance(dims, torch.Tensor):
        dims = dims.tolist()
    dims = list(dims)
    if len(dims) != ndim:
        raise ValueError(f"dims length {len(dims)} must equal tensor ndim {ndim}")
    norm = []
    for d in dims:
        if d < 0:
            d += ndim
        if not (0 <= d < ndim):
            raise ValueError(f"dimension out of range: {d}")
        norm.append(d)
    if sorted(norm) != list(range(ndim)):
        raise ValueError(f"dims must be a permutation of [0..{ndim - 1}], got {norm}")
    return norm


def permute_copy(x: torch.Tensor, dims):
    logger.debug("ILUVATAR GEMS PERMUTE_COPY")

    rank = x.dim()
    if rank == 0:
        return x.clone()

    dims = _normalize_dims(dims, rank)
    out_shape = [x.size(d) for d in dims]
    n_elements = int(
        torch.tensor(out_shape, dtype=torch.int64).prod().item()
        if len(out_shape) > 0
        else 1
    )

    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)

    # Early exit for zero elements
    if n_elements == 0:
        return out

    NDIMS = x.dim()
    if NDIMS == 0:
        out.copy_(x)
        return out

    out_shape_t = torch.tensor(out_shape, device=x.device, dtype=torch.int64)
    in_strides_t = torch.tensor(x.stride(), device=x.device, dtype=torch.int64)
    out_strides_t = torch.tensor(out.stride(), device=x.device, dtype=torch.int64)
    perm_t = torch.tensor(dims, device=x.device, dtype=torch.int64)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(x.device):
        _permute_copy_kernel[grid](
            x,
            out,
            n_elements,
            out_shape_t,
            in_strides_t,
            out_strides_t,
            perm_t,
            NDIMS=NDIMS,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return out


def permute_copy_out(x: torch.Tensor, dims, out: torch.Tensor):
    logger.debug("ILUVATAR GEMS PERMUTE_COPY_OUT")

    if x.dtype != out.dtype:
        raise RuntimeError("permute_copy_out: dtype mismatch between input and output.")

    rank = x.dim()
    if rank == 0:
        out.copy_(x)
        return out

    dims = _normalize_dims(dims, rank)
    expected_shape = [x.size(d) for d in dims]

    if list(out.shape) != expected_shape:
        raise RuntimeError(
            f"permute_copy_out: output shape {out.shape} does not match expected {expected_shape}."
        )

    if x.numel() == 0:
        return out

    n_elements = out.numel()
    if n_elements == 0:
        return out

    NDIMS = x.dim()

    out_shape_t = torch.tensor(expected_shape, device=x.device, dtype=torch.int64)
    in_strides_t = torch.tensor(x.stride(), device=x.device, dtype=torch.int64)
    out_strides_t = torch.tensor(out.stride(), device=x.device, dtype=torch.int64)
    perm_t = torch.tensor(dims, device=x.device, dtype=torch.int64)

    BLOCK_SIZE = 2048
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(x.device):
        _permute_copy_kernel[grid](
            x,
            out,
            n_elements,
            out_shape_t,
            in_strides_t,
            out_strides_t,
            perm_t,
            NDIMS=NDIMS,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return out