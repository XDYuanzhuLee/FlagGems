import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger("flag_gems." + __name__)


def adaptive_pool2d_output_size(in_h, in_w, out_h, out_w):
    return out_h, out_w


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 16}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_stages=3, num_warps=8),
    ],
    key=["out_h", "out_w", "in_h", "in_w"],
)
@triton.jit
def adaptive_max_pool2d_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    # Input tensor strides
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    # Input/Output shapes
    in_n,
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    # Tiling meta-parameters
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    max_acc = tl.full((BLOCK_H, BLOCK_W), float("-inf"), dtype=tl.float32)
    indices_acc = tl.full((BLOCK_H, BLOCK_W), -1, dtype=tl.int64)

    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    # Compute the kernel size for each output position
    # For adaptive pooling, the kernel size varies based on output position
    start_h = h_out_offsets * in_h // out_h
    end_h = (h_out_offsets + 1) * in_h // out_h
    start_w = w_out_offsets * in_w // out_w
    end_w = (w_out_offsets + 1) * in_w // out_w

    # Iterate through all input positions
    for h_in in range(in_h):
        for w_in in range(in_w):
            # Compute linear index for this input position
            current_indices = tl.full((BLOCK_H, BLOCK_W), h_in * in_w + w_in, dtype=tl.int64)

            # Check if this input position contributes to each output position
            h_valid = (h_in >= start_h[:, None]) & (h_in < end_h[:, None])
            w_valid = (w_in >= start_w[None, :]) & (w_in < end_w[None, :])
            valid = h_valid & w_valid

            # Load the value
            input_offset = h_in * in_stride_h + w_in * in_stride_w
            current_val = tl.load(input_base_ptr + input_offset)

            # Update max and indices where valid and current > max
            comp = current_val > max_acc
            update = valid & comp
            max_acc = tl.where(update, current_val, max_acc)
            indices_acc = tl.where(update, current_indices, indices_acc)

    out_base_ptr = output_ptr + pid_nc * out_h * out_w
    out_indices_base_ptr = indices_ptr + pid_nc * out_h * out_w

    out_h_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    output_block_ptr = out_base_ptr + out_h_offsets[:, None] * out_w + out_w_offsets[None, :]
    indices_block_ptr = out_indices_base_ptr + out_h_offsets[:, None] * out_w + out_w_offsets[None, :]

    out_mask = (out_h_offsets[:, None] < out_h) & (out_w_offsets[None, :] < out_w)
    tl.store(
        output_block_ptr, max_acc.to(output_ptr.type.element_ty), mask=out_mask
    )
    tl.store(
        indices_block_ptr, indices_acc, mask=out_mask
    )


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 16}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 16}, num_stages=3, num_warps=8),
    ],
    key=["out_h", "out_w", "in_h", "in_w"],
)
@triton.jit
def adaptive_max_pool2d_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    # Input/Output shapes
    in_n,
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    # Strides
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    # Tiling meta-parameters
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    num_w_blocks = tl.cdiv(in_w, BLOCK_W)
    h_block_idx = pid_hw // num_w_blocks
    w_block_idx = pid_hw % num_w_blocks
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    grad_input_base_ptr = grad_input_ptr + n_idx * in_stride_n + c_idx * in_stride_h
    grad_output_base_ptr = grad_output_ptr + n_idx * out_stride_n + c_idx * out_stride_h

    h_in_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_in_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    grad_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Compute the output positions that contribute to each input position
    start_h = h_in_offsets * out_h // in_h
    end_h = (h_in_offsets + 1) * out_h // in_h
    start_w = w_in_offsets * out_w // in_w
    end_w = (w_in_offsets + 1) * out_w // in_w

    for h_out in range(out_h):
        for w_out in range(out_w):
            h_valid = (h_out >= start_h[:, None]) & (h_out < end_h[:, None])
            w_valid = (w_out >= start_w[None, :]) & (w_out < end_w[None, :])
            valid = h_valid & w_valid

            grad_out_ptr = (
                grad_output_base_ptr + h_out * out_stride_h + w_out * out_stride_w
            )
            grad_out_val = tl.load(grad_out_ptr)
            grad_acc += tl.where(valid, grad_out_val, 0.0)

    grad_input_store_ptr = (
        grad_input_base_ptr
        + h_in_offsets[:, None] * in_stride_h
        + w_in_offsets[None, :] * in_stride_w
    )
    in_write_mask = (h_in_offsets[:, None] < in_h) & (w_in_offsets[None, :] < in_w)
    tl.store(
        grad_input_store_ptr,
        grad_acc.to(grad_input_ptr.type.element_ty),
        mask=in_write_mask,
    )


def adaptive_max_pool2d(input: torch.Tensor, output_size):
    logger.debug("ILUVATAR GEMS ADAPTIVE_MAX_POOL2D")

    if input.dim() != 4:
        raise ValueError(
            f"expected 4D input (got {input.dim()}D input)"
        )

    in_n, in_c, in_h, in_w = input.shape

    if isinstance(output_size, int):
        out_h = out_w = output_size
    elif isinstance(output_size, (tuple, list)):
        if len(output_size) == 1:
            out_h = out_w = output_size[0]
        elif len(output_size) == 2:
            out_h, out_w = output_size
        else:
            raise ValueError(
                f"output_size must have 2 elements (got {len(output_size)})"
            )
    else:
        raise TypeError(
            f"output_size must be int or tuple of ints (got {type(output_size)})"
        )

    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"output_size must be greater than 0 (got {out_h}, {out_w})")

    input = input.contiguous()

    output = torch.empty(
        (in_n, in_c, out_h, out_w), device=input.device, dtype=input.dtype
    )
    indices = torch.empty(
        (in_n, in_c, out_h, out_w), device=input.device, dtype=torch.int64
    )

    if output.numel() == 0:
        return output, indices

    grid = lambda meta: (
        in_n * in_c,
        triton.cdiv(out_h, meta["BLOCK_H"]) * triton.cdiv(out_w, meta["BLOCK_W"]),
    )

    adaptive_max_pool2d_kernel[grid](
        input,
        output,
        indices,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        in_n,
        in_c,
        in_h,
        in_w,
        out_h,
        out_w,
    )

    return output, indices


def adaptive_max_pool2d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    indices: torch.Tensor,
):
    logger.debug("ILUVATAR GEMS ADAPTIVE_MAX_POOL2D BACKWARD")

    if grad_output.dim() != 4:
        raise ValueError(
            f"expected 4D grad_output (got {grad_output.dim()}D input)"
        )

    in_n, in_c, in_h, in_w = input.shape
    out_h, out_w = grad_output.shape[2], grad_output.shape[3]

    grad_output = grad_output.contiguous()

    grad_input = torch.zeros_like(input, dtype=torch.float32)

    if grad_output.numel() == 0:
        return grad_input.to(grad_output.dtype)

    grid = lambda meta: (
        in_n * in_c,
        triton.cdiv(in_h, meta["BLOCK_H"]) * triton.cdiv(in_w, meta["BLOCK_W"]),
    )

    adaptive_max_pool2d_backward_kernel[grid](
        grad_output,
        indices,
        grad_input,
        in_n,
        in_c,
        in_h,
        in_w,
        out_h,
        out_w,
        grad_input.stride(0),
        grad_input.stride(1),
        grad_input.stride(2),
        grad_input.stride(3),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_output.stride(3),
    )

    return grad_input.to(grad_output.dtype)