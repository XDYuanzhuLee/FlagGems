import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle

rsqrt = tl_extra_shim.rsqrt

logger = logging.getLogger("flag_gems." + __name__)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def batch_norm_no_update_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    running_mean_ptr,
    running_var_ptr,
    output_ptr,
    mean_ptr,
    inv_std_ptr,
    batch_dim,
    spatial_dim,
    input_batch_stride,
    input_feat_stride,
    input_spatial_stride,
    output_batch_stride,
    output_feat_stride,
    output_spatial_stride,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    feat_pid = tle.program_id(0)

    # Compute mean and var from input
    mean = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    var = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    cnt = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    m_num_steps = tl.cdiv(batch_dim, BLOCK_M)
    n_num_steps = tl.cdiv(spatial_dim, BLOCK_N)

    for m_step in range(0, m_num_steps):
        for n_step in range(0, n_num_steps):
            spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
            spatial_mask = spatial_offset < spatial_dim

            batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
            batch_mask = batch_offset < batch_dim

            curr_input_ptr = (
                input_ptr
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )

            mask = batch_mask[:, None] & spatial_mask[None, :]
            curr_input = tl.load(curr_input_ptr, mask=mask).to(tl.float32)

            step = m_step * n_num_steps + n_step + 1
            new_mean = tl.where(mask, mean + (curr_input - mean) / step, mean)
            new_var = tl.where(
                mask, var + (curr_input - new_mean) * (curr_input - mean), var
            )
            cnt += mask.to(tl.int32)
            mean = new_mean
            var = new_var

    final_mean = tl.sum(mean * cnt) / (batch_dim * spatial_dim)
    var = tl.sum(var + cnt * (mean - final_mean) * (mean - final_mean)) / (
        batch_dim * spatial_dim
    )
    inv_std = rsqrt(var + eps)
    mean = final_mean

    # Store mean and inv_std (for backward pass) - NOT updating running stats
    tl.store(feat_pid + mean_ptr, mean)
    tl.store(feat_pid + inv_std_ptr, inv_std)

    # Load weight and bias
    if weight_ptr:
        weight = tl.load(feat_pid + weight_ptr).to(tl.float32)
    else:
        weight = 1.0
    if bias_ptr:
        bias = tl.load(feat_pid + bias_ptr).to(tl.float32)
    else:
        bias = 0.0

    # Compute output
    for m_step in range(0, tl.cdiv(batch_dim, BLOCK_M)):
        for n_step in range(0, tl.cdiv(spatial_dim, BLOCK_N)):
            batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
            batch_mask = batch_offset < batch_dim

            spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
            spatial_mask = spatial_offset < spatial_dim

            curr_input_ptr = (
                input_ptr
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )
            curr_output_ptr = (
                output_ptr
                + output_feat_stride * feat_pid
                + output_batch_stride * batch_offset[:, None]
                + output_spatial_stride * spatial_offset[None, :]
            )

            curr_input = tl.load(
                curr_input_ptr, mask=batch_mask[:, None] & spatial_mask[None, :]
            ).to(tl.float32)
            output = weight * (curr_input - mean) * inv_std + bias

            tl.store(
                curr_output_ptr,
                output,
                mask=batch_mask[:, None] & spatial_mask[None, :],
            )


def make_3d_for_bn(input: torch.Tensor) -> torch.Tensor:
    """
    Converts the input to a 3D view for batch normalization.
    """
    if input.ndim == 2:
        input = input.unsqueeze(-1)
    elif input.ndim >= 4:
        input = input.flatten(2, -1)
    return input


def _batch_norm_no_update(
    input: torch.Tensor,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    momentum=0.1,
    eps=1e-05,
):
    logger.debug("METAX GEMS BATCH_NORM_NO_UPDATE")

    input_3d = make_3d_for_bn(input)

    batch_dim, feat_dim, spatial_dim = input_3d.shape
    output = torch.empty_like(input_3d)

    mean = torch.empty(feat_dim, device=input.device, dtype=input.dtype)
    inv_std = torch.empty(feat_dim, device=input.device, dtype=input.dtype)

    with torch_device_fn.device(input.device):
        batch_norm_no_update_kernel[(feat_dim,)](
            input_3d,
            weight,
            bias,
            running_mean,
            running_var,
            output,
            mean,
            inv_std,
            batch_dim,
            spatial_dim,
            *input_3d.stride(),
            *output.stride(),
            eps,
        )

    # Return format: (output, mean, inv_std, reserved)
    # reserved is kept for compatibility with aten operator
    reserved = torch.empty(0, dtype=torch.uint8, device=input.device)
    return output.view_as(input), mean, inv_std, reserved