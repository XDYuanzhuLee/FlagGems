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


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_OH": 4, "BLOCK_OW": 4}, num_warps=4),
        triton.Config({"BLOCK_OH": 8, "BLOCK_OW": 4}, num_warps=4),
        triton.Config({"BLOCK_OH": 4, "BLOCK_OW": 8}, num_warps=4),
        triton.Config({"BLOCK_OH": 8, "BLOCK_OW": 8}, num_warps=8),
        triton.Config({"BLOCK_OH": 16, "BLOCK_OW": 8}, num_warps=8),
        triton.Config({"BLOCK_OH": 8, "BLOCK_OW": 16}, num_warps=8),
        triton.Config({"BLOCK_OH": 16, "BLOCK_OW": 16}, num_warps=8),
    ],
    key=["OH", "OW"],
)
@triton.jit
def adaptive_max_pool2d_kernel(
    inp_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    OH,
    OW,
    stride_n,
    stride_c,
    stride_h,
    stride_w,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    # Grid: (OH_blocks * OW_blocks, N * C)
    # OH_blocks = ceil(OH / BLOCK_OH), OW_blocks = ceil(OW / BLOCK_OW)

    pid_hw = tle.program_id(0)
    pid_nc = tle.program_id(1)

    # Compute (n, c) from pid_nc
    n_idx = pid_nc // C
    c_idx = pid_nc % C

    # Compute (oh_block, ow_block) from pid_hw
    num_w_blocks = tl.cdiv(OW, BLOCK_OW)
    oh_block = pid_hw // num_w_blocks
    ow_block = pid_hw % num_w_blocks

    # Get input base pointer for this (n, c)
    inp_base = n_idx * stride_n + c_idx * stride_c

    # Initialize output array for this block
    dtype = inp_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    output = tl.zeros((BLOCK_OH, BLOCK_OW), dtype=tl.float32)

    # Compute output position offsets for this block
    oh_offsets = oh_block * BLOCK_OH + tl.arange(0, BLOCK_OH)
    ow_offsets = ow_block * BLOCK_OW + tl.arange(0, BLOCK_OW)

    # For each output position in the block, compute max
    # We iterate over all output positions in the block
    for local_oh_idx in range(BLOCK_OH):
        global_oh = oh_block * BLOCK_OH + local_oh_idx
        if global_oh >= OH:
            break

        for local_ow_idx in range(BLOCK_OW):
            global_ow = ow_block * BLOCK_OW + local_ow_idx
            if global_ow >= OW:
                break

            # Compute input range for this output position
            ih_start = global_oh * H // OH
            ih_end = (global_oh + 1) * H // OH
            if ih_end > H:
                ih_end = H

            iw_start = global_ow * W // OW
            iw_end = (global_ow + 1) * W // OW
            if iw_end > W:
                iw_end = W

            # Find max in this input range
            local_max = min_val
            for ih in range(ih_start, ih_end):
                row_ptr = inp_base + ih * stride_h
                for iw in range(iw_start, iw_end):
                    val = tl.load(row_ptr + iw * stride_w)
                    local_max = tl.max(local_max, val)

            # Store at the correct position in output array
            # Use a mask to only update valid positions
            output = tl.where(
                (local_oh_idx < BLOCK_OH) & (local_ow_idx < BLOCK_OW),
                local_max,
                output
            )

    # Store output - only valid positions
    out_base = pid_nc * OH * OW
    out_mask = (oh_offsets[:, None] < OH) & (ow_offsets[None, :] < OW)
    out_h_offsets = oh_offsets[:, None]
    out_w_offsets = ow_offsets[None, :]

    output_ptr = out_ptr + out_base + out_h_offsets * OW + out_w_offsets
    tl.store(output_ptr, output, mask=out_mask)


def adaptive_max_pool2d(inp, output_size):
    logger.debug("METAX GEMS ADAPTIVE_MAX_POOL2D")
    N, C, H, W = inp.shape
    if isinstance(output_size, int):
        OH, OW = output_size, output_size
    else:
        OH, OW = output_size[0], output_size[1]

    out = torch.empty((N, C, OH, OW), dtype=inp.dtype, device=inp.device)

    if out.numel() == 0:
        return out, torch.empty((N, C, OH, OW), dtype=torch.long, device=inp.device)

    # Grid: (OH_blocks * OW_blocks, N * C)
    # First dimension: spatial blocks, Second dimension: batch * channel
    grid = (
        triton.cdiv(OH, 8) * triton.cdiv(OW, 8),
        N * C,
    )

    with torch_device_fn.device(inp.device):
        adaptive_max_pool2d_kernel[grid](
            inp,
            out,
            N,
            C,
            H,
            W,
            OH,
            OW,
            inp.stride(0),
            inp.stride(1),
            inp.stride(2),
            inp.stride(3),
        )

    # Return dummy indices for compatibility
    indices = torch.empty((N, C, OH, OW), dtype=torch.long, device=inp.device)
    return out, indices