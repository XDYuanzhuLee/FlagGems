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


def max_pool3d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    ceil_mode: bool = False,
) -> int:
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    numerator = in_size + 2 * padding - effective_kernel_size
    if ceil_mode:
        output_size = (numerator + stride - 1) // stride + 1
        # PyTorch-compatible adjustment for ceil_mode
        if (output_size - 1) * stride >= in_size + padding:
            output_size -= 1
    else:
        output_size = numerator // stride + 1

    return output_size


def _parse_pool_params_3d(kernel_size, stride, padding, dilation):
    """Parse pooling parameters for 3D pooling."""

    def _parse_param(param, name, default=None):
        if param is None:
            return default
        if isinstance(param, int):
            return param, param, param
        if isinstance(param, (list, tuple)) and len(param) == 3:
            return param
        raise ValueError(f"Invalid {name}: {param}")

    kernel_d, kernel_h, kernel_w = _parse_param(kernel_size, "kernel_size")
    stride_d, stride_h, stride_w = _parse_param(
        stride, "stride", default=(kernel_d, kernel_h, kernel_w)
    )
    padding_d, padding_h, padding_w = _parse_param(padding, "padding", default=(0, 0, 0))
    dilation_d, dilation_h, dilation_w = _parse_param(
        dilation, "dilation", default=(1, 1, 1)
    )

    if stride_d <= 0 or stride_h <= 0 or stride_w <= 0:
        raise ValueError(
            f"stride must be positive, but got stride=({stride_d}, {stride_h}, {stride_w})"
        )
    if padding_d < 0 or padding_h < 0 or padding_w < 0:
        raise ValueError(
            f"padding must be non-negative, but got padding=({padding_d}, {padding_h}, {padding_w})"
        )
    if dilation_d <= 0 or dilation_h <= 0 or dilation_w <= 0:
        raise ValueError(
            f"dilation must be positive, but got dilation=({dilation_d}, {dilation_h}, {dilation_w})"
        )

    return (
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        dilation_d,
        dilation_h,
        dilation_w,
    )


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_IN_D": 4, "BLOCK_IN_H": 8, "BLOCK_IN_W": 8}, num_warps=4),
        triton.Config({"BLOCK_IN_D": 8, "BLOCK_IN_H": 8, "BLOCK_IN_W": 8}, num_warps=4),
        triton.Config({"BLOCK_IN_D": 8, "BLOCK_IN_H": 16, "BLOCK_IN_W": 8}, num_warps=4),
        triton.Config({"BLOCK_IN_D": 8, "BLOCK_IN_H": 8, "BLOCK_IN_W": 16}, num_warps=4),
        triton.Config({"BLOCK_IN_D": 16, "BLOCK_IN_H": 8, "BLOCK_IN_W": 8}, num_warps=8),
    ],
    key=["in_d", "in_h", "in_w", "kernel_d", "kernel_h", "kernel_w", "stride_d", "stride_h", "stride_w"],
)
@triton.jit
def max_pool3d_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    # Shape info
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    # Strides for grad_output/indices
    out_stride_nc,
    out_stride_d,
    out_stride_h,
    out_stride_w,
    # Pooling parameters
    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_d: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_d: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    # Tiling parameters
    BLOCK_IN_D: tl.constexpr,
    BLOCK_IN_H: tl.constexpr,
    BLOCK_IN_W: tl.constexpr,
):
    # Grid: (N*C, num_d_blocks * num_h_blocks * num_w_blocks)
    pid_nc = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    num_w_blocks = tl.cdiv(in_w, BLOCK_IN_W)
    num_h_blocks = tl.cdiv(in_h, BLOCK_IN_H)
    num_d_blocks = tl.cdiv(in_d, BLOCK_IN_D)

    d_block_idx = pid_dhw // (num_h_blocks * num_w_blocks)
    h_block_idx = (pid_dhw // num_w_blocks) % num_h_blocks
    w_block_idx = pid_dhw % num_w_blocks

    # Compute input positions this block handles
    d_in_offsets = d_block_idx * BLOCK_IN_D + tl.arange(0, BLOCK_IN_D)
    h_in_offsets = h_block_idx * BLOCK_IN_H + tl.arange(0, BLOCK_IN_H)
    w_in_offsets = w_block_idx * BLOCK_IN_W + tl.arange(0, BLOCK_IN_W)

    # Flat index into input (for comparing with indices)
    current_input_flat_idx = (
        d_in_offsets[:, None, None] * in_h * in_w
        + h_in_offsets[None, :, None] * in_w
        + w_in_offsets[None, None, :]
    )

    # Accumulator for gradients
    grad_acc = tl.zeros((BLOCK_IN_D, BLOCK_IN_H, BLOCK_IN_W), dtype=tl.float32)

    # Base pointers for this N*C slice
    n_idx = pid_nc
    indices_base_ptr = indices_ptr + n_idx * out_stride_nc
    grad_output_base_ptr = grad_output_ptr + n_idx * out_stride_nc

    # For each kernel position, check if any output position maps to current input
    for kd in tl.static_range(0, kernel_d):
        for kh in tl.static_range(0, kernel_h):
            for kw in tl.static_range(0, kernel_w):
                # Compute the output position that this kernel position maps to
                # Start from input position, subtract padding and kernel offset
                numerator_d = d_in_offsets[:, None, None] + padding_d - kd * dilation_d
                numerator_h = h_in_offsets[None, :, None] + padding_h - kh * dilation_h
                numerator_w = w_in_offsets[None, None, :] + padding_w - kw * dilation_w

                # Check if division yields integer (i.e., this output position is valid)
                valid_map_mask_d = numerator_d % stride_d == 0
                valid_map_mask_h = numerator_h % stride_h == 0
                valid_map_mask_w = numerator_w % stride_w == 0
                valid_map_mask = valid_map_mask_d & valid_map_mask_h & valid_map_mask_w

                d_out = numerator_d // stride_d
                h_out = numerator_h // stride_h
                w_out = numerator_w // stride_w

                # Check if output position is within bounds
                out_bounds_mask = (
                    (d_out >= 0)
                    & (d_out < out_d)
                    & (h_out >= 0)
                    & (h_out < out_h)
                    & (w_out >= 0)
                    & (w_out < out_w)
                )
                load_mask = valid_map_mask & out_bounds_mask

                # Compute output offset
                out_offsets = (
                    d_out * out_stride_d
                    + h_out * out_stride_h
                    + w_out * out_stride_w
                )

                # Load indices and check if they match current input flat index
                indices_block = tl.load(
                    indices_base_ptr + out_offsets, mask=load_mask, other=-1
                )
                match_mask = indices_block == current_input_flat_idx

                # Load gradient and accumulate
                grad_block = tl.load(
                    grad_output_base_ptr + out_offsets, mask=match_mask, other=0.0
                )
                grad_acc += grad_block

    # Store the result
    grad_input_base_ptr = grad_input_ptr + n_idx * in_d * in_h * in_w
    grad_input_offsets = (
        d_in_offsets[:, None, None] * in_h * in_w
        + h_in_offsets[None, :, None] * in_w
        + w_in_offsets[None, None, :]
    )

    store_mask = (
        (d_in_offsets[:, None, None] < in_d)
        & (h_in_offsets[None, :, None] < in_h)
        & (w_in_offsets[None, None, :] < in_w)
    )

    tl.store(grad_input_base_ptr + grad_input_offsets, grad_acc, mask=store_mask)


def max_pool3d_with_indices_backward(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode: bool,
    indices: torch.Tensor,
):
    logger.debug("METAX GEMS MAX_POOL3D_WITH_INDICES_BACKWARD")

    grad_output = grad_output.contiguous()
    indices = indices.contiguous()

    params = _parse_pool_params_3d(kernel_size, stride, padding, dilation)
    (
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        dilation_d,
        dilation_h,
        dilation_w,
    ) = params

    in_n, in_c, in_d, in_h, in_w = self.shape
    out_d, out_h, out_w = grad_output.shape[2], grad_output.shape[3], grad_output.shape[4]

    grad_input = torch.zeros_like(self, dtype=torch.float32)

    if grad_input.numel() == 0:
        return grad_input.to(grad_output.dtype)

    grid = lambda meta: (
        in_n * in_c,
        triton.cdiv(in_d, meta["BLOCK_IN_D"])
        * triton.cdiv(in_h, meta["BLOCK_IN_H"])
        * triton.cdiv(in_w, meta["BLOCK_IN_W"]),
    )

    out_stride_nc = out_d * out_h * out_w
    out_stride_d = out_h * out_w
    out_stride_h = out_w
    out_stride_w = 1

    with torch_device_fn.device(self.device):
        max_pool3d_backward_kernel[grid](
            grad_output,
            indices,
            grad_input,
            in_d,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            out_stride_nc,
            out_stride_d,
            out_stride_h,
            out_stride_w,
            kernel_d,
            kernel_h,
            kernel_w,
            stride_d,
            stride_h,
            stride_w,
            padding_d,
            padding_h,
            padding_w,
            dilation_d,
            dilation_h,
            dilation_w,
        )

    return grad_input.to(grad_output.dtype)