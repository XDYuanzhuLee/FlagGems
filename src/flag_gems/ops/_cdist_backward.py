import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _cdist_backward_p2_kernel(
    grad_ptr,
    x1_ptr,
    x2_ptr,
    cdist_ptr,
    out_ptr,
    N,
    P,
    M,
    stride_x1n,
    stride_x2p,
    stride_cn,
    stride_outn,
    BLOCK_M: tl.constexpr,
):
    """Backward kernel for cdist with p=2 (Euclidean distance)."""
    pid_n = tle.program_id(0)

    x1_offset = pid_n * stride_x1n
    x1_row_ptr = x1_ptr + x1_offset

    m_offsets = tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    x1_vals = tl.load(x1_row_ptr + m_offsets, mask=m_mask, other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for j in range(P):
        x2_offset = j * stride_x2p
        x2_row_ptr = x2_ptr + x2_offset
        x2_vals = tl.load(x2_row_ptr + m_offsets, mask=m_mask, other=0.0).to(tl.float32)

        diff = x1_vals - x2_vals

        cdist_idx = pid_n * stride_cn + j
        g = tl.load(grad_ptr + cdist_idx).to(tl.float32)
        c = tl.load(cdist_ptr + cdist_idx).to(tl.float32)

        safe_c = tl.where(c != 0.0, c, 1.0)
        scale = g / safe_c

        contrib = diff * scale
        acc += contrib

    out_offset = pid_n * stride_outn
    out_ptr_row = out_ptr + out_offset
    tl.store(out_ptr_row + m_offsets, acc.to(out_ptr.dtype.element_ty), mask=m_mask)


@libentry()
@triton.jit
def _cdist_backward_p1_kernel(
    grad_ptr,
    x1_ptr,
    x2_ptr,
    out_ptr,
    N,
    P,
    M,
    stride_x1n,
    stride_x2p,
    stride_cn,
    stride_outn,
    BLOCK_M: tl.constexpr,
):
    """Backward kernel for cdist with p=1 (L1 distance)."""
    pid_n = tle.program_id(0)

    x1_offset = pid_n * stride_x1n
    x1_row_ptr = x1_ptr + x1_offset

    m_offsets = tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    x1_vals = tl.load(x1_row_ptr + m_offsets, mask=m_mask, other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for j in range(P):
        x2_offset = j * stride_x2p
        x2_row_ptr = x2_ptr + x2_offset
        x2_vals = tl.load(x2_row_ptr + m_offsets, mask=m_mask, other=0.0).to(tl.float32)

        diff = x1_vals - x2_vals
        sign = tl.where(diff > 0, 1.0, tl.where(diff < 0, -1.0, 0.0))

        cdist_idx = pid_n * stride_cn + j
        g = tl.load(grad_ptr + cdist_idx).to(tl.float32)

        acc += g * sign

    out_offset = pid_n * stride_outn
    out_ptr_row = out_ptr + out_offset
    tl.store(out_ptr_row + m_offsets, acc.to(out_ptr.dtype.element_ty), mask=m_mask)


@libentry()
@triton.jit
def _cdist_backward_inf_kernel(
    grad_ptr,
    x1_ptr,
    x2_ptr,
    cdist_ptr,
    out_ptr,
    N,
    P,
    M,
    stride_x1n,
    stride_x2p,
    stride_cn,
    stride_outn,
    BLOCK_M: tl.constexpr,
):
    """Backward kernel for cdist with p=inf (Chebyshev distance)."""
    pid_n = tle.program_id(0)

    x1_offset = pid_n * stride_x1n
    x1_row_ptr = x1_ptr + x1_offset

    m_offsets = tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    x1_vals = tl.load(x1_row_ptr + m_offsets, mask=m_mask, other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for j in range(P):
        x2_offset = j * stride_x2p
        x2_row_ptr = x2_ptr + x2_offset
        x2_vals = tl.load(x2_row_ptr + m_offsets, mask=m_mask, other=0.0).to(tl.float32)

        diff = x1_vals - x2_vals
        abs_diff = tl.abs(diff)

        cdist_idx = pid_n * stride_cn + j
        g = tl.load(grad_ptr + cdist_idx).to(tl.float32)
        c = tl.load(cdist_ptr + cdist_idx).to(tl.float32)

        is_max = tl.where(abs_diff == c, 1.0, 0.0)
        sign = tl.where(diff > 0, 1.0, tl.where(diff < 0, -1.0, 0.0))

        acc += g * sign * is_max

    out_offset = pid_n * stride_outn
    out_ptr_row = out_ptr + out_offset
    tl.store(out_ptr_row + m_offsets, acc.to(out_ptr.dtype.element_ty), mask=m_mask)


@libentry()
@triton.jit
def _cdist_backward_general_kernel(
    grad_ptr,
    x1_ptr,
    x2_ptr,
    cdist_ptr,
    out_ptr,
    N,
    P,
    M,
    p_val,
    stride_x1n,
    stride_x2p,
    stride_cn,
    stride_outn,
    BLOCK_M: tl.constexpr,
):
    """Backward kernel for cdist with general p value."""
    pid_n = tle.program_id(0)

    x1_offset = pid_n * stride_x1n
    x1_row_ptr = x1_ptr + x1_offset

    m_offsets = tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    x1_vals = tl.load(x1_row_ptr + m_offsets, mask=m_mask, other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for j in range(P):
        x2_offset = j * stride_x2p
        x2_row_ptr = x2_ptr + x2_offset
        x2_vals = tl.load(x2_row_ptr + m_offsets, mask=m_mask, other=0.0).to(tl.float32)

        diff = x1_vals - x2_vals
        abs_diff = tl.abs(diff)
        sign = tl.where(diff > 0, 1.0, tl.where(diff < 0, -1.0, 0.0))

        cdist_idx = pid_n * stride_cn + j
        g = tl.load(grad_ptr + cdist_idx).to(tl.float32)
        c = tl.load(cdist_ptr + cdist_idx).to(tl.float32)

        safe_c = tl.where(c == 0.0, 1.0, c)
        ratio = abs_diff / safe_c

        safe_ratio = tl.where(ratio == 0.0, 0.0, ratio)
        log_ratio = tl.where(safe_ratio > 0.0, tl.log(safe_ratio), 0.0)
        powered = tl.where(safe_ratio > 0.0, tl.exp((p_val - 1.0) * log_ratio), 0.0)

        contrib = tl.where(c == 0.0, 0.0, g * sign * powered)
        acc += contrib

    out_offset = pid_n * stride_outn
    out_ptr_row = out_ptr + out_offset
    tl.store(out_ptr_row + m_offsets, acc.to(out_ptr.dtype.element_ty), mask=m_mask)


def _cdist_backward(grad, x1, x2, p, cdist):
    """Compute gradient of cdist forward pass with respect to x1."""
    logger.debug("GEMS _CDIST_BACKWARD")

    assert x1.ndim == 3, "cdist requires 3D input (batch, N, M)"
    assert x2.ndim == 3, "cdist requires 3D input (batch, P, M)"
    assert x1.shape[0] == x2.shape[0], "Batch sizes must match"
    assert x1.shape[2] == x2.shape[2], "Feature dimensions must match"

    batch = x1.shape[0]
    N = x1.shape[1]
    M = x1.shape[2]
    P = x2.shape[1]

    assert batch == 1, "Only batch=1 supported currently"

    grad = grad.contiguous()
    x1 = x1.contiguous()
    x2 = x2.contiguous()
    cdist = cdist.contiguous()

    out = torch.empty_like(x1)

    BLOCK_M = min(triton.next_power_of_2(M), 1024)
    grid = (N,)

    with torch_device_fn.device(x1.device):
        if p == 2.0:
            _cdist_backward_p2_kernel[grid](
                grad,
                x1,
                x2,
                cdist,
                out,
                N,
                P,
                M,
                x1.stride(1),
                x2.stride(1),
                cdist.stride(1),
                out.stride(1),
                BLOCK_M=BLOCK_M,
            )
        elif p == 1.0:
            _cdist_backward_p1_kernel[grid](
                grad,
                x1,
                x2,
                out,
                N,
                P,
                M,
                x1.stride(1),
                x2.stride(1),
                cdist.stride(1),
                out.stride(1),
                BLOCK_M=BLOCK_M,
            )
        elif math.isinf(p):
            _cdist_backward_inf_kernel[grid](
                grad,
                x1,
                x2,
                cdist,
                out,
                N,
                P,
                M,
                x1.stride(1),
                x2.stride(1),
                cdist.stride(1),
                out.stride(1),
                BLOCK_M=BLOCK_M,
            )
        else:
            _cdist_backward_general_kernel[grid](
                grad,
                x1,
                x2,
                cdist,
                out,
                N,
                P,
                M,
                p,
                x1.stride(1),
                x2.stride(1),
                cdist.stride(1),
                out.stride(1),
                BLOCK_M=BLOCK_M,
            )

    return out