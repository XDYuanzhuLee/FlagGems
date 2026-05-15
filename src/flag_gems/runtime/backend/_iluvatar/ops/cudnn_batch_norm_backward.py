import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)
rsqrt = tl_extra_shim.rsqrt


def make_3d_for_bn(input: Tensor) -> Tensor:
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
@triton.autotune(
    configs=runtime.get_tuned_config("batch_norm"),
    key=["batch_dim", "spatial_dim"],
)
@triton.heuristics(runtime.get_heuristic_config("batch_norm"))
@triton.jit
def cudnn_batch_norm_backward_kernel(
    output_grad_pointer,
    input_pointer,
    mean_pointer,
    inv_std_pointer,
    weight_pointer,
    input_grad_pointer,
    weight_grad_pointer,
    bias_grad_pointer,
    batch_dim,
    spatial_dim,
    output_grad_batch_stride,
    output_grad_feat_stride,
    output_grad_spatial_stride,
    input_batch_stride,
    input_feat_stride,
    input_spatial_stride,
    input_grad_batch_stride,
    input_grad_feat_stride,
    input_grad_spatial_stride,
    input_grad_mask: tl.constexpr,
    weight_grad_mask: tl.constexpr,
    bias_grad_mask: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    feat_pid = tl.program_id(axis=0)

    mean = tl.load(feat_pid + mean_pointer).to(tl.float32)
    inv_std = tl.load(feat_pid + inv_std_pointer).to(tl.float32)

    term1 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    term2 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for m_step in range(0, tl.cdiv(batch_dim, BLOCK_M)):
        for n_step in range(0, tl.cdiv(spatial_dim, BLOCK_N)):
            batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
            batch_mask = batch_offset < batch_dim

            spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
            spatial_mask = spatial_offset < spatial_dim

            curr_output_grad_pointer = (
                output_grad_pointer
                + output_grad_feat_stride * feat_pid
                + output_grad_batch_stride * batch_offset[:, None]
                + output_grad_spatial_stride * spatial_offset[None, :]
            )
            curr_input_pointer = (
                input_pointer
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )

            mask = batch_mask[:, None] & spatial_mask[None, :]
            curr_input = tl.load(curr_input_pointer, mask=mask).to(tl.float32)

            curr_pre_lin = (curr_input - mean) * inv_std
            curr_output_grad = tl.load(curr_output_grad_pointer, mask=mask).to(
                tl.float32
            )

            term1 += curr_pre_lin * curr_output_grad
            term2 += curr_output_grad

    term1 = tl.sum(term1)
    term2 = tl.sum(term2)

    if weight_grad_mask:
        tl.store(feat_pid + weight_grad_pointer, term1)
    if bias_grad_mask:
        tl.store(feat_pid + bias_grad_pointer, term2)

    if not input_grad_mask:
        return

    if weight_pointer:
        weight = tl.load(feat_pid + weight_pointer).to(tl.float32)
    else:
        weight = 1.0

    count = batch_dim * spatial_dim

    for m_step in range(0, tl.cdiv(batch_dim, BLOCK_M)):
        for n_step in range(0, tl.cdiv(spatial_dim, BLOCK_N)):
            batch_offset = m_step * BLOCK_M + tl.arange(0, BLOCK_M)
            batch_mask = batch_offset < batch_dim

            spatial_offset = n_step * BLOCK_N + tl.arange(0, BLOCK_N)
            spatial_mask = spatial_offset < spatial_dim

            curr_output_grad_pointer = (
                output_grad_pointer
                + output_grad_feat_stride * feat_pid
                + output_grad_batch_stride * batch_offset[:, None]
                + output_grad_spatial_stride * spatial_offset[None, :]
            )
            curr_input_pointer = (
                input_pointer
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )
            curr_input_grad_pointer = (
                input_grad_pointer
                + input_grad_feat_stride * feat_pid
                + input_grad_batch_stride * batch_offset[:, None]
                + input_grad_spatial_stride * spatial_offset[None, :]
            )

            curr_input = tl.load(
                curr_input_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]
            ).to(tl.float32)
            curr_pre_lin = (curr_input - mean) * inv_std
            curr_output_grad = tl.load(
                curr_output_grad_pointer,
                mask=batch_mask[:, None] & spatial_mask[None, :],
            ).to(tl.float32)
            curr_input_grad = (
                inv_std
                * weight
                * (curr_output_grad - (term1 * curr_pre_lin + term2) / count)
            )
            tl.store(
                curr_input_grad_pointer,
                curr_input_grad,
                mask=batch_mask[:, None] & spatial_mask[None, :],
            )


def cudnn_batch_norm_backward(
    grad_output: Tensor,
    input: Tensor,
    weight: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    save_mean: Tensor,
    save_invstd: Tensor,
    epsilon: float,
    reserveSpace: Tensor,
):
    """
    Backward pass for cudnn_batch_norm.

    Args:
        grad_output: Gradient of the loss w.r.t. the output of batch norm
        input: Input tensor
        weight: Weight tensor (gamma)
        running_mean: Running mean (not used in backward for training)
        running_var: Running variance (not used in backward for training)
        save_mean: Saved mean from forward pass
        save_invstd: Saved inverse std from forward pass
        epsilon: Epsilon value for numerical stability
        reserveSpace: Reserve space from forward pass (not used)

    Returns:
        Tuple of (grad_input, grad_weight, grad_bias)
    """
    logger.debug("ILUVATAR GEMS CUDNN_BATCH_NORM_BACKWARD")

    input_3d = make_3d_for_bn(input)
    output_grad_3d = make_3d_for_bn(grad_output)

    batch_dim, feat_dim, spatial_dim = input_3d.shape

    input_grad = torch.empty_like(input_3d)
    weight_grad = torch.empty((feat_dim,), dtype=input.dtype, device=input.device)
    bias_grad = torch.empty((feat_dim,), dtype=input.dtype, device=input.device)

    # Launches 1D grid where each program operates over one feature.
    with torch_device_fn.device(input.device):
        cudnn_batch_norm_backward_kernel[(feat_dim,)](
            output_grad_3d,
            input_3d,
            save_mean,
            save_invstd,
            weight,
            input_grad,
            weight_grad,
            bias_grad,
            batch_dim,
            spatial_dim,
            *output_grad_3d.stride(),
            *input_3d.stride(),
            *input_grad.stride(),
            True,  # input_grad_mask
            True,  # weight_grad_mask
            True,  # bias_grad_mask
        )

    # Pads output with None because a gradient is necessary for
    # all input arguments.
    return (
        input_grad.view_as(input),
        weight_grad,
        bias_grad,
    )