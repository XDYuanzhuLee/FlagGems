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
def affine_grid_generator_kernel(
    theta_ptr,
    output_ptr,
    N,
    H,
    W,
    align_corners_i32,  # 1 for True, 0 for False
    stride_theta_n,
    stride_theta_h,
    stride_theta_w,
    stride_out_n,
    stride_out_h,
    stride_out_w,
    stride_out_c,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Args:
        theta: (N, 2, 3) affine transformation matrices
        output: (N, H, W, 2) sampling grid
        N: batch size
        H: output height
        W: output width
        align_corners: if True, corners are aligned, otherwise not
    """
    pid = tle.program_id(0)
    num_elements = N * H * W
    if pid >= tl.cdiv(num_elements, BLOCK_SIZE):
        return

    elements_offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = elements_offset < num_elements

    # Decode flat index to (n, h, w)
    n = elements_offset // (H * W)
    h = (elements_offset % (H * W)) // W
    w = elements_offset % W

    # Load theta for this batch
    theta_base = n * stride_theta_n

    # Load the affine transformation matrix
    # theta shape: (N, 2, 3)
    # theta[n, 0, :] = [a00, a01, a02]
    # theta[n, 1, :] = [a10, a11, a12]
    a00 = tl.load(theta_ptr + theta_base + 0 * stride_theta_h + 0 * stride_theta_w)
    a01 = tl.load(theta_ptr + theta_base + 0 * stride_theta_h + 1 * stride_theta_w)
    a02 = tl.load(theta_ptr + theta_base + 0 * stride_theta_h + 2 * stride_theta_w)
    a10 = tl.load(theta_ptr + theta_base + 1 * stride_theta_h + 0 * stride_theta_w)
    a11 = tl.load(theta_ptr + theta_base + 1 * stride_theta_h + 1 * stride_theta_w)
    a12 = tl.load(theta_ptr + theta_base + 1 * stride_theta_h + 2 * stride_theta_w)

    # Calculate normalized coordinates
    # x and y are the pixel coordinates in the output grid
    w_f32 = w.to(tl.float32)
    h_f32 = h.to(tl.float32)

    # Compute normalized coordinates for align_corners=True
    # x_norm = 2 * w / (W - 1) - 1, for W=1 we set to 0
    W_minus_1_f32 = tl.cast(W - 1, tl.float32)
    H_minus_1_f32 = tl.cast(H - 1, tl.float32)
    x_norm_true = tl.where(
        W > 1,
        2.0 * w_f32 / W_minus_1_f32 - 1.0,
        tl.zeros_like(w_f32)
    )
    y_norm_true = tl.where(
        H > 1,
        2.0 * h_f32 / H_minus_1_f32 - 1.0,
        tl.zeros_like(h_f32)
    )

    # Compute normalized coordinates for align_corners=False
    # x_norm = 2 * (w + 0.5) / W - 1 = 2*w/W - 1 + 1/W
    W_f32 = tl.cast(W, tl.float32)
    H_f32 = tl.cast(H, tl.float32)
    x_norm_false = w_f32 * 2.0 / W_f32 - 1.0 + 1.0 / W_f32
    y_norm_false = h_f32 * 2.0 / H_f32 - 1.0 + 1.0 / H_f32

    # Select based on align_corners flag
    x_norm = tl.where(align_corners_i32 == 1, x_norm_true, x_norm_false)
    y_norm = tl.where(align_corners_i32 == 1, y_norm_true, y_norm_false)

    # Apply affine transformation
    # output[x, y, 0] = a00 * x + a01 * y + a02
    # output[x, y, 1] = a10 * x + a11 * y + a12
    output_x = a00 * x_norm + a01 * y_norm + a02
    output_y = a10 * x_norm + a11 * y_norm + a12

    # Store results
    out_base = n * stride_out_n + h * stride_out_h + w * stride_out_w
    tl.store(output_ptr + out_base + 0 * stride_out_c, output_x, mask=mask)
    tl.store(output_ptr + out_base + 1 * stride_out_c, output_y, mask=mask)


def affine_grid_generator(theta: torch.Tensor, size: list, align_corners: bool = False):
    """
    Generates a 2D sampling grid from an affine transformation matrix.

    Args:
        theta: (N, 2, 3) tensor of affine transformation matrices
        size: List of 4 integers [N, C, H, W]
        align_corners: If True, the corner pixels of the input and output
                      are aligned.

    Returns:
        grid: (N, H, W, 2) tensor of normalized coordinates
    """
    logger.debug("METAX GEMS AFFINE_GRID_GENERATOR")
    N, C, H, W = size
    assert theta.shape == (N, 2, 3), f"Expected theta shape ({N}, 2, 3), got {theta.shape}"

    # Output shape: (N, H, W, 2)
    output = torch.empty((N, H, W, 2), device=theta.device, dtype=theta.dtype)

    num_elements = N * H * W
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(theta.device):
        affine_grid_generator_kernel[grid](
            theta,
            output,
            N,
            H,
            W,
            1 if align_corners else 0,
            theta.stride(0),
            theta.stride(1),
            theta.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return output