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

    Args:
        input: Input to render 3D.

    Returns:
        Input's 3D view.
    """
    if input.ndim == 2:
        input = input.unsqueeze(-1)

    elif input.ndim >= 4:
        input = input.flatten(2, -1)

    return input


@libentry()
@triton.heuristics(
    {
        "BLOCK_M": ["tl.cdiv(batch_dim, 32)", "batch_dim"],
        "BLOCK_N": ["tl.cdiv(spatial_dim, 32)", "spatial_dim"],
    }
)
@triton.jit
def native_batch_norm_legit_no_training_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    running_mean_pointer,
    running_var_pointer,
    output_pointer,
    mean_pointer,
    inv_std_pointer,
    batch_dim,
    spatial_dim,
    feat_dim,
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
    # Each program processes one feature dimension
    feat_pid = tle.program_id(0)

    # Load running mean and inv_std (which is 1/sqrt(var + eps))
    mean = tl.load(feat_pid + running_mean_pointer).to(tl.float32)
    running_var = tl.load(feat_pid + running_var_pointer).to(tl.float32)
    inv_std = rsqrt(running_var + eps)

    # Load weight and bias if present
    if weight_pointer:
        weight = tl.load(feat_pid + weight_pointer).to(tl.float32)
    else:
        weight = 1.0
    if bias_pointer:
        bias = tl.load(feat_pid + bias_pointer).to(tl.float32)
    else:
        bias = 0.0

    # Store mean and inv_std (for backward compatibility)
    tl.store(feat_pid + mean_pointer, mean)
    tl.store(feat_pid + inv_std_pointer, inv_std)

    # Normalize and denormalize the input
    for m_step in range(0, tl.cdiv(batch_dim, BLOCK_M)):
        for n_step in range(0, tl.cdiv(spatial_dim, BLOCK_N)):
            batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
            batch_mask = batch_offset < batch_dim

            spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
            spatial_mask = spatial_offset < spatial_dim

            curr_input_pointer = (
                input_pointer
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )
            curr_output_pointer = (
                output_pointer
                + output_feat_stride * feat_pid
                + output_batch_stride * batch_offset[:, None]
                + output_spatial_stride * spatial_offset[None, :]
            )

            curr_input = tl.load(
                curr_input_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]
            ).to(tl.float32)
            # Normalize: (x - mean) * inv_std
            # Denormalize: normalized * weight + bias
            output = weight * (curr_input - mean) * inv_std + bias

            tl.store(
                curr_output_pointer,
                output,
                mask=batch_mask[:, None] & spatial_mask[None, :],
            )


def native_batch_norm_legit_no_training(
    input: torch.Tensor,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    momentum=0.1,
    eps=1e-05,
):
    logger.debug("METAX GEMS NATIVE_BATCH_NORM_LEGIT_NO_TRAINING")

    # Convert input to 3D for batch normalization
    input_3d = make_3d_for_bn(input)

    batch_dim, feat_dim, spatial_dim = input_3d.shape
    output = torch.empty_like(input_3d)

    # Allocate output mean and inv_std buffers
    mean = torch.empty(feat_dim, device=input.device, dtype=input.dtype)
    inv_std = torch.empty(feat_dim, device=input.device, dtype=input.dtype)

    # Ensure contiguous memory layout
    if weight is not None:
        weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    if running_mean is not None:
        running_mean = running_mean.contiguous()
    if running_var is not None:
        running_var = running_var.contiguous()

    # Launch kernel: 1D grid where each program operates over one feature
    with torch_device_fn.device(input.device):
        native_batch_norm_legit_no_training_kernel[(feat_dim,)](
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
            feat_dim,
            *input_3d.stride(),
            *output.stride(),
            eps,
        )

    # Return output reshaped to original input shape, empty mean/var (compatibility)
    return output.view_as(input), mean, inv_std