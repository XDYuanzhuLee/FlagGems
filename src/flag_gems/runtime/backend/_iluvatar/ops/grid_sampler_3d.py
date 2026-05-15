import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry

logger = logging.getLogger("flag_gems." + __name__)


def _register_grid_sampler_3d():
    """Register grid_sampler_3d to aten library on import"""
    try:
        aten_lib = torch.library.Library("aten", "IMPL")
        aten_lib.impl("grid_sampler_3d", grid_sampler_3d, "CUDA")
    except Exception as e:
        logger.debug(f"grid_sampler_3d registration skipped: {e}")


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("grid_sampler_3d"),
    key=["N", "C", "OD", "OH", "OW", "ID", "IH", "IW"],
)
@triton.jit
def grid_sampler_3d_kernel(
    output_ptr,
    input_ptr,
    grid_ptr,
    N,
    C,
    OD,
    OH,
    OW,
    ID,
    IH,
    IW,
    interpolation_mode,
    padding_mode,
    align_corners,
    BLOCK_SIZE: tl.constexpr,
):
    # output: (N, C, OD, OH, OW)
    # input: (N, C, ID, IH, IW)
    # grid: (N, OD, OH, OW, 3)

    pid = tl.program_id(0)
    num_od_oh_ow = OD * OH * OW
    num_output_elements = N * C * OD * OH * OW

    if pid * BLOCK_SIZE >= num_output_elements:
        return

    # Calculate output indices
    output_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = output_idx < num_output_elements

    # Calculate N, C, OD, OH, OW indices
    n = output_idx // (C * num_od_oh_ow)
    c_od_oh_ow = output_idx % (C * num_od_oh_ow)
    c = c_od_oh_ow // num_od_oh_ow
    od_oh_ow = c_od_oh_ow % num_od_oh_ow
    od = od_oh_ow // (OH * OW)
    oh_ow = od_oh_ow % (OH * OW)
    oh = oh_ow // OW
    ow = oh_ow % OW

    # Load grid coordinates: (N, OD, OH, OW, 3)
    grid_base = n * (OD * OH * OW * 3) + od * (OH * OW * 3) + oh * (OW * 3) + ow * 3
    gx = tl.load(grid_ptr + grid_base + 0, mask=mask)
    gy = tl.load(grid_ptr + grid_base + 1, mask=mask)
    gz = tl.load(grid_ptr + grid_base + 2, mask=mask)

    # Transform grid coordinates to pixel space
    if align_corners:
        # For align_corners=True, the grid range [-1, 1] maps to [0, D-1], etc.
        x = (gx + 1.0) * 0.5 * (IW - 1)
        y = (gy + 1.0) * 0.5 * (IH - 1)
        z = (gz + 1.0) * 0.5 * (ID - 1)
    else:
        # For align_corners=False, the grid range [-1, 1] maps to [-0.5, D-0.5], etc.
        x = (gx + 1.0) * 0.5 * IW - 0.5
        y = (gy + 1.0) * 0.5 * IH - 0.5
        z = (gz + 1.0) * 0.5 * ID - 0.5

    # Clamp based on padding_mode
    if padding_mode == 0:  # zeros
        # Clamp to valid range, out of bounds will be zero
        x_clamped = tl.minimum(tl.maximum(x, -1.0), tl.constexpr(IW - 1))
        y_clamped = tl.minimum(tl.maximum(y, -1.0), tl.constexpr(IH - 1))
        z_clamped = tl.minimum(tl.maximum(z, -1.0), tl.constexpr(ID - 1))
    elif padding_mode == 1:  # border
        x_clamped = tl.minimum(tl.maximum(x, 0.0), tl.constexpr(IW - 1))
        y_clamped = tl.minimum(tl.maximum(y, 0.0), tl.constexpr(IH - 1))
        z_clamped = tl.minimum(tl.maximum(z, 0.0), tl.constexpr(ID - 1))
    else:  # reflection
        # Clamp and reflect
        x_clamped = tl.minimum(tl.maximum(x, -1.0), tl.constexpr(IW - 1))
        y_clamped = tl.minimum(tl.maximum(y, -1.0), tl.constexpr(IH - 1))
        z_clamped = tl.minimum(tl.maximum(z, -1.0), tl.constexpr(ID - 1))

    # Get integer and fractional parts
    x0 = tl.floor(x_clamped).to(tl.int32)
    y0 = tl.floor(y_clamped).to(tl.int32)
    z0 = tl.floor(z_clamped).to(tl.int32)

    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # For nearest neighbor, just use the nearest integer
    if interpolation_mode == 1:  # nearest
        x_nearest = tl.where(x_clamped < 0, x0 - 1, x0)
        y_nearest = tl.where(y_clamped < 0, y0 - 1, y0)
        z_nearest = tl.where(z_clamped < 0, z0 - 1, z0)

        x_nearest = tl.minimum(tl.maximum(x_nearest, 0), tl.constexpr(IW - 1))
        y_nearest = tl.minimum(tl.maximum(y_nearest, 0), tl.constexpr(IH - 1))
        z_nearest = tl.minimum(tl.maximum(z_nearest, 0), tl.constexpr(ID - 1))

        # Load value at nearest neighbor
        input_offset = (
            n * (C * ID * IH * IW)
            + c * (ID * IH * IW)
            + z_nearest * (IH * IW)
            + y_nearest * IW
            + x_nearest
        )
        result = tl.load(input_ptr + input_offset, mask=mask)
    else:  # bilinear (interpolation_mode == 0)
        # Compute interpolation weights
        xf = x_clamped - tl.floor(x_clamped)
        yf = y_clamped - tl.floor(y_clamped)
        zf = z_clamped - tl.floor(z_clamped)

        # Clamp indices to valid range
        x0_clamped = tl.minimum(tl.maximum(x0, 0), tl.constexpr(IW - 1))
        x1_clamped = tl.minimum(tl.maximum(x1, 0), tl.constexpr(IW - 1))
        y0_clamped = tl.minimum(tl.maximum(y0, 0), tl.constexpr(IH - 1))
        y1_clamped = tl.minimum(tl.maximum(y1, 0), tl.constexpr(IH - 1))
        z0_clamped = tl.minimum(tl.maximum(z0, 0), tl.constexpr(ID - 1))
        z1_clamped = tl.minimum(tl.maximum(z1, 0), tl.constexpr(ID - 1))

        # Load 8 corners
        input_base = n * (C * ID * IH * IW) + c * (ID * IH * IW)

        # z0 layer
        offset_z0_y0_x0 = input_base + z0_clamped * (IH * IW) + y0_clamped * IW + x0_clamped
        offset_z0_y0_x1 = input_base + z0_clamped * (IH * IW) + y0_clamped * IW + x1_clamped
        offset_z0_y1_x0 = input_base + z0_clamped * (IH * IW) + y1_clamped * IW + x0_clamped
        offset_z0_y1_x1 = input_base + z0_clamped * (IH * IW) + y1_clamped * IW + x1_clamped

        # z1 layer
        offset_z1_y0_x0 = input_base + z1_clamped * (IH * IW) + y0_clamped * IW + x0_clamped
        offset_z1_y0_x1 = input_base + z1_clamped * (IH * IW) + y0_clamped * IW + x1_clamped
        offset_z1_y1_x0 = input_base + z1_clamped * (IH * IW) + y1_clamped * IW + x0_clamped
        offset_z1_y1_x1 = input_base + z1_clamped * (IH * IW) + y1_clamped * IW + x1_clamped

        c000 = tl.load(input_ptr + offset_z0_y0_x0, mask=mask)
        c001 = tl.load(input_ptr + offset_z0_y0_x1, mask=mask)
        c010 = tl.load(input_ptr + offset_z0_y1_x0, mask=mask)
        c011 = tl.load(input_ptr + offset_z0_y1_x1, mask=mask)
        c100 = tl.load(input_ptr + offset_z1_y0_x0, mask=mask)
        c101 = tl.load(input_ptr + offset_z1_y0_x1, mask=mask)
        c110 = tl.load(input_ptr + offset_z1_y1_x0, mask=mask)
        c111 = tl.load(input_ptr + offset_z1_y1_x1, mask=mask)

        # Trilinear interpolation
        # First interpolate in x direction
        c00 = c000 * (1 - xf) + c001 * xf
        c01 = c010 * (1 - xf) + c011 * xf
        c10 = c100 * (1 - xf) + c101 * xf
        c11 = c110 * (1 - xf) + c111 * xf

        # Then interpolate in y direction
        c0 = c00 * (1 - yf) + c01 * yf
        c1 = c10 * (1 - yf) + c11 * yf

        # Finally interpolate in z direction
        result = c0 * (1 - zf) + c1 * zf

    tl.store(output_ptr + output_idx, result, mask=mask)


def grid_sampler_3d(
    input: torch.Tensor,
    grid: torch.Tensor,
    interpolation_mode: int = 0,
    padding_mode: int = 0,
    align_corners: bool = False,
) -> torch.Tensor:
    logger.debug("ILUVATAR GEMS grid_sampler_3d")

    N, C, ID, IH, IW = input.shape
    grid_N, OD, OH, OW, grid_dim = grid.shape

    assert N == grid_N, "Batch size mismatch between input and grid"
    assert grid_dim == 3, "Grid must have 3 dimensions (x, y, z)"

    output = torch.empty((N, C, OD, OH, OW), device=input.device, dtype=input.dtype)

    # Calculate grid
    def grid_fn(meta):
        total_elements = N * C * OD * OH * OW
        return ((total_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    grid_sampler_3d_kernel[grid_fn](
        output,
        input,
        grid,
        N,
        C,
        OD,
        OH,
        OW,
        ID,
        IH,
        IW,
        interpolation_mode,
        padding_mode,
        align_corners,
    )

    return output


# Auto-register when module is imported
_register_grid_sampler_3d()