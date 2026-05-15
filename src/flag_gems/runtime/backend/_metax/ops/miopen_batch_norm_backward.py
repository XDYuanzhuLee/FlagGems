import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle

rsqrt = tl_extra_shim.rsqrt

logger = logging.getLogger("flag_gems." + __name__)


def make_3d_for_bn(input: torch.Tensor) -> torch.Tensor:
    """
    Converts the input to a 3D view for batch normalization.
    """
    if input.ndim == 2:
        input = input.unsqueeze(-1)
    elif input.ndim >= 4:
        input = input.flatten(2, -1)
    return input


@libentry()
@triton.jit
def miopen_batch_norm_backward_kernel(
    grad_output_ptr,
    input_ptr,
    weight_ptr,
    save_mean_ptr,
    save_invstd_ptr,
    grad_input_ptr,
    grad_weight_ptr,
    grad_bias_ptr,
    batch_dim,
    spatial_dim,
    feat_dim,
    grad_output_batch_stride,
    grad_output_feat_stride,
    grad_output_spatial_stride,
    input_batch_stride,
    input_feat_stride,
    input_spatial_stride,
    grad_input_batch_stride,
    grad_input_feat_stride,
    grad_input_spatial_stride,
    grad_weight_mask: tl.constexpr,
    grad_bias_mask: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    feat_pid = tle.program_id(0)

    mean = tl.load(feat_pid + save_mean_ptr).to(tl.float32)
    inv_std = tl.load(feat_pid + save_invstd_ptr).to(tl.float32)

    term1 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    term2 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for m_step in range(0, tl.cdiv(batch_dim, BLOCK_M)):
        for n_step in range(0, tl.cdiv(spatial_dim, BLOCK_N)):
            batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
            batch_mask = batch_offset < batch_dim

            spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
            spatial_mask = spatial_offset < spatial_dim

            curr_grad_output_ptr = (
                grad_output_ptr
                + grad_output_feat_stride * feat_pid
                + grad_output_batch_stride * batch_offset[:, None]
                + grad_output_spatial_stride * spatial_offset[None, :]
            )
            curr_input_ptr = (
                input_ptr
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )

            mask = batch_mask[:, None] & spatial_mask[None, :]
            curr_input = tl.load(curr_input_ptr, mask=mask).to(tl.float32)

            curr_pre_lin = (curr_input - mean) * inv_std
            curr_grad_output = tl.load(curr_grad_output_ptr, mask=mask).to(
                tl.float32
            )

            term1 += curr_pre_lin * curr_grad_output
            term2 += curr_grad_output

    term1 = tl.sum(term1)
    term2 = tl.sum(term2)

    if grad_weight_mask:
        tl.store(feat_pid + grad_weight_ptr, term1)
    if grad_bias_mask:
        tl.store(feat_pid + grad_bias_ptr, term2)

    if weight_ptr:
        weight = tl.load(feat_pid + weight_ptr).to(tl.float32)
    else:
        weight = 1.0

    count = batch_dim * spatial_dim

    for m_step in range(0, tl.cdiv(batch_dim, BLOCK_M)):
        for n_step in range(0, tl.cdiv(spatial_dim, BLOCK_N)):
            batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
            batch_mask = batch_offset < batch_dim

            spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
            spatial_mask = spatial_offset < spatial_dim

            curr_grad_output_ptr = (
                grad_output_ptr
                + grad_output_feat_stride * feat_pid
                + grad_output_batch_stride * batch_offset[:, None]
                + grad_output_spatial_stride * spatial_offset[None, :]
            )
            curr_input_ptr = (
                input_ptr
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )
            curr_grad_input_ptr = (
                grad_input_ptr
                + grad_input_feat_stride * feat_pid
                + grad_input_batch_stride * batch_offset[:, None]
                + grad_input_spatial_stride * spatial_offset[None, :]
            )

            curr_input = tl.load(
                curr_input_ptr, mask=batch_mask[:, None] & spatial_mask[None, :]
            ).to(tl.float32)
            curr_pre_lin = (curr_input - mean) * inv_std
            curr_grad_output = tl.load(
                curr_grad_output_ptr,
                mask=batch_mask[:, None] & spatial_mask[None, :],
            ).to(tl.float32)
            curr_grad_input = (
                inv_std
                * weight
                * (curr_grad_output - (term1 * curr_pre_lin + term2) / count)
            )
            tl.store(
                curr_grad_input_ptr,
                curr_grad_input,
                mask=batch_mask[:, None] & spatial_mask[None, :],
            )


def miopen_batch_norm_backward(
    grad_output,
    input,
    weight,
    running_mean,
    running_var,
    save_mean,
    save_invstd,
    epsilon=1e-05,
    *args,
):
    """
    MIOpen batch norm backward operator.
    This is similar to native_batch_norm_backward but uses the MIOpen operator schema.

    Note: This function accepts additional optional arguments (*args) for compatibility
    with the native_batch_norm_backward benchmark (train, output_mask), but ignores them
    as the MIOpen version always computes all gradients.
    """
    logger.debug("METAX GEMS MIOPEN_BATCH_NORM_BACKWARD")

    # Make 3d for batch normalization
    input_3d = make_3d_for_bn(input)
    grad_output_3d = make_3d_for_bn(grad_output)

    batch_dim, feat_dim, spatial_dim = input_3d.shape

    # Allocate outputs
    grad_input = torch.empty_like(input_3d)
    grad_weight = torch.empty((feat_dim,), dtype=input.dtype, device=input.device)
    grad_bias = torch.empty((feat_dim,), dtype=input.dtype, device=input.device)

    # Launches 1D grid where each program operates over one feature.
    # Use fixed block sizes for better performance and to avoid 32-bit overflow
    # on Metax (which has pointer size limitations)
    BLOCK_M = 16 if batch_dim <= 32 else (32 if batch_dim <= 128 else 64)
    BLOCK_N = 32 if spatial_dim <= 64 else (64 if spatial_dim <= 256 else 128)

    with torch_device_fn.device(input.device):
        miopen_batch_norm_backward_kernel[(feat_dim,)](
            grad_output_3d,
            input_3d,
            weight,
            save_mean,
            save_invstd,
            grad_input,
            grad_weight,
            grad_bias,
            batch_dim,
            spatial_dim,
            feat_dim,
            *grad_output_3d.stride(),
            *input_3d.stride(),
            *grad_input.stride(),
            weight is not None,
            weight is not None,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

    # Resize to original shape
    return (
        grad_input.view_as(input),
        grad_weight,
        grad_bias,
    )