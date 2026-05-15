import logging

import torch

from flag_gems import ops

logger = logging.getLogger(__name__)


def miopen_batch_norm(
    input: torch.Tensor,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    training=False,
    exponential_average_factor=0.1,
    epsilon=1e-5,
):
    r"""
    Applies Batch Normalization over a 4D input (NCHW) or 5D input (NCDHW).

    This is the Iluvatar specialized version of miopen_batch_norm.
    When MIOpen is not available, it delegates to the native batch_norm implementation.

    Args:
        input: Input tensor of shape (N, C, H, W) or (N, C, D, H, W)
        weight: Optional tensor of shape (C,) for scale
        bias: Optional tensor of shape (C,) for bias
        running_mean: Optional running mean tensor of shape (C,)
        running_var: Optional running variance tensor of shape (C,)
        training: Whether in training mode
        exponential_average_factor: Momentum factor for updating running statistics
        epsilon: Small value added to variance for numerical stability

    Returns:
        Tuple of (output, save_mean, save_var)
    """
    logger.debug("ILUVATAR GEMS MIOPEN_BATCH_NORM")

    if running_mean is None:
        running_mean = input.new_zeros(input.shape[1])
    if running_var is None:
        running_var = input.new_ones(input.shape[1])

    # Convert exponential_average_factor to momentum
    # momentum = 1 - exponential_average_factor
    momentum = 1.0 - exponential_average_factor

    # Delegate to the native batch_norm implementation
    output, save_mean, inv_std = ops.batch_norm(
        input=input,
        weight=weight,
        bias=bias,
        running_mean=running_mean,
        running_var=running_var,
        training=training,
        momentum=momentum,
        eps=epsilon,
    )

    # Return save_mean and save_var instead of inv_std
    # save_var is computed as 1 / inv_std^2 = inv_std^(-2)
    save_var = torch.reciprocal(inv_std * inv_std)

    return output, save_mean, save_var


def miopen_batch_norm_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight=None,
    running_mean=None,
    running_var=None,
    save_mean=None,
    save_var=None,
    epsilon=1e-5,
    output_mask=(True, True, True),
):
    r"""
    Backward pass for miopen_batch_norm.

    This is the Iluvatar specialized version of miopen_batch_norm_backward.

    Args:
        grad_output: Gradient of the output
        input: Input tensor
        weight: Optional weight tensor
        running_mean: Optional running mean
        running_var: Optional running variance
        save_mean: Saved mean from forward pass
        save_var: Saved variance from forward pass
        epsilon: Small value for numerical stability
        output_mask: Tuple indicating which gradients to compute

    Returns:
        Tuple of (input_grad, weight_grad, bias_grad)
    """
    logger.debug("ILUVATAR GEMS MIOPEN_BATCH_NORM_BACKWARD")

    if save_mean is None or save_var is None:
        raise ValueError(
            "save_mean and save_var must be provided for backward pass"
        )

    # Convert save_var to inv_std for the backward function
    save_invstd = torch.reciprocal(torch.sqrt(save_var + epsilon))

    # Delegate to the native batch_norm_backward implementation
    input_grad, weight_grad, bias_grad = ops.batch_norm_backward(
        grad_out=grad_output,
        input=input,
        weight=weight,
        running_mean=running_mean,
        running_var=running_var,
        save_mean=save_mean,
        save_invstd=save_invstd,
        train=True,
        eps=epsilon,
        output_mask=output_mask,
    )

    return input_grad, weight_grad, bias_grad