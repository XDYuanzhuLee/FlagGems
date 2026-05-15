import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, True, True],
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit
def binary_cross_entropy_backward_func(grad_output, pred, target):
    """
    Element-wise gradient of binary cross entropy loss w.r.t. input predictions.
    grad = grad_output * (pred - target) / (pred * (1 - pred))
    """
    # Compute (pred - target) / (pred * (1 - pred))
    numerator = pred - target
    denominator = pred * (1.0 - pred)
    # Avoid division by zero
    safe_denominator = tl.where(denominator != 0.0, denominator, 1.0)
    grad = numerator / safe_denominator
    # Handle cases where pred is 0 or 1 (where gradient should be 0)
    grad = tl.where((pred <= 0.0) | (pred >= 1.0), 0.0, grad)
    return grad_output * grad


@pointwise_dynamic(
    is_tensor=[True, True, True, True],
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit
def binary_cross_entropy_backward_weighted_func(grad_output, pred, target, weight):
    """
    Element-wise gradient of binary cross entropy loss w.r.t. input predictions with weight.
    grad = grad_output * weight * (pred - target) / (pred * (1 - pred))
    """
    numerator = pred - target
    denominator = pred * (1.0 - pred)
    safe_denominator = tl.where(denominator != 0.0, denominator, 1.0)
    grad = numerator / safe_denominator
    grad = tl.where((pred <= 0.0) | (pred >= 1.0), 0.0, grad)
    return grad_output * weight * grad


def binary_cross_entropy_backward(grad_output, pred, target, weight=None, reduction=1):
    """
    Backward pass for binary cross entropy loss.

    Args:
        grad_output: Gradient from the loss
        pred: Input predictions (same shape as target)
        target: Target labels (same shape as pred)
        weight: Optional weight tensor
        reduction: 0=none, 1=mean, 2=sum

    Returns:
        Gradient w.r.t. pred input
    """
    logger.debug("GEMS binary_cross_entropy_backward")

    # Handle empty tensors
    if pred.numel() == 0:
        return torch.empty_like(pred)

    # Determine which kernel to use
    if weight is not None:
        grad = binary_cross_entropy_backward_weighted_func(
            grad_output, pred, target, weight
        )
    else:
        grad = binary_cross_entropy_backward_func(grad_output, pred, target)

    # Apply reduction
    if reduction == 0:  # NONE
        return grad
    elif reduction == 1:  # MEAN
        return grad / pred.numel()
    elif reduction == 2:  # SUM - same as NONE for backward
        return grad
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")