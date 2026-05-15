import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def replication_pad3d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    N,
    C,
    D_out,
    H_out,
    W_out,
    D_in,
    H_in,
    W_in,
    pad_w_before,
    pad_w_after,
    pad_h_before,
    pad_h_after,
    pad_d_before,
    pad_d_after,
    grad_output_stride_n,
    grad_output_stride_c,
    grad_output_stride_d,
    grad_output_stride_h,
    grad_output_stride_w,
    grad_input_stride_n,
    grad_input_stride_c,
    grad_input_stride_d,
    grad_input_stride_h,
    grad_input_stride_w,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Unravel linear indices into (n, c, d_out, h_out, w_out)
    w_out = offs % W_out
    tmp = offs // W_out
    h_out = tmp % H_out
    tmp = tmp // H_out
    d_out = tmp % D_out
    tmp = tmp // D_out
    c = tmp % C
    n = tmp // C

    # Compute clamped input indices (this is the backward mapping)
    w_in = w_out - pad_w_before
    w_in = tl.maximum(w_in, 0)
    w_in = tl.minimum(w_in, W_in - 1)

    h_in = h_out - pad_h_before
    h_in = tl.maximum(h_in, 0)
    h_in = tl.minimum(h_in, H_in - 1)

    d_in = d_out - pad_d_before
    d_in = tl.maximum(d_in, 0)
    d_in = tl.minimum(d_in, D_in - 1)

    # Compute input and output pointers (strided)
    grad_output_offset = (
        n * grad_output_stride_n
        + c * grad_output_stride_c
        + d_out * grad_output_stride_d
        + h_out * grad_output_stride_h
        + w_out * grad_output_stride_w
    )
    grad_input_offset = (
        n * grad_input_stride_n
        + c * grad_input_stride_c
        + d_in * grad_input_stride_d
        + h_in * grad_input_stride_h
        + w_in * grad_input_stride_w
    )

    # Load gradient from output and atomically add to input
    grad_val = tl.load(grad_output_ptr + grad_output_offset, mask=mask, other=0)
    tl.atomic_add(grad_input_ptr + grad_input_offset, grad_val, mask=mask)


def _normalize_3d_pad(padding):
    if isinstance(padding, (list, tuple)) and len(padding) == 6:
        return tuple(int(x) for x in padding)
    raise ValueError(
        "padding must be a sequence of 6 integers: (pad_w_before, pad_w_after, pad_h_before, pad_h_after, pad_d_before, pad_d_after)"
    )


def _get_5d_shape_and_strides(t: torch.Tensor):
    if t.dim() == 5:
        N, C, D, H, W = t.shape
        sN, sC, sD, sH, sW = t.stride()
        was_4d = False
        return (N, C, D, H, W), (sN, sC, sD, sH, sW), was_4d
    elif t.dim() == 4:
        C, D, H, W = t.shape
        sC, sD, sH, sW = t.stride()
        N = 1
        sN = 0
        was_4d = True
        return (N, C, D, H, W), (sN, sC, sD, sH, sW), was_4d
    else:
        raise ValueError("Input must be 4D (C, D, H, W) or 5D (N, C, D, H, W).")


def replication_pad3d_backward(grad_output, self, padding):
    logger.debug("ILUVATAR GEMS REPLICATION_PAD3D_BACKWARD")

    (
        pad_w_before,
        pad_w_after,
        pad_h_before,
        pad_h_after,
        pad_d_before,
        pad_d_after,
    ) = _normalize_3d_pad(padding)

    # Get shapes
    (
        (N, C, D_out, H_out, W_out),
        (go_sN, go_sC, go_sD, go_sH, go_sW),
        _,
    ) = _get_5d_shape_and_strides(grad_output)

    D_in = D_out - pad_d_before - pad_d_after
    H_in = H_out - pad_h_before - pad_h_after
    W_in = W_out - pad_w_before - pad_w_after

    # Create output gradient tensor
    grad_input = torch.zeros(
        (N, C, D_in, H_in, W_in),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )

    n_elements = grad_output.numel()
    if n_elements == 0:
        return grad_input

    # Compute input strides
    gi_sN = grad_input.stride(0)
    gi_sC = grad_input.stride(1)
    gi_sD = grad_input.stride(2)
    gi_sH = grad_input.stride(3)
    gi_sW = grad_input.stride(4)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    replication_pad3d_backward_kernel[grid](
        grad_output,
        grad_input,
        N,
        C,
        D_out,
        H_out,
        W_out,
        D_in,
        H_in,
        W_in,
        pad_w_before,
        pad_w_after,
        pad_h_before,
        pad_h_after,
        pad_d_before,
        pad_d_after,
        go_sN,
        go_sC,
        go_sD,
        go_sH,
        go_sW,
        gi_sN,
        gi_sC,
        gi_sD,
        gi_sH,
        gi_sW,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return grad_input