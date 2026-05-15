import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def binary_cross_entropy_with_logits_func(logits, targets):
    # Numerically stable computation of binary cross entropy
    # BCE = max(x, 0) - x * target + log(1 + exp(-|x|))
    # For numerical stability with different dtypes, use float32 for exp/log
    logits_fp32 = logits.to(tl.float32)
    targets_fp32 = targets.to(tl.float32)

    abs_logits = tl.abs(logits_fp32)
    max_term = tl.maximum(logits_fp32, 0.0)
    loss_fp32 = max_term - logits_fp32 * targets_fp32 + tl.log(1.0 + tl.exp(-abs_logits))

    # Convert back to original dtype
    loss = loss_fp32.to(logits.type.scalar)
    return loss


@pointwise_dynamic(is_tensor=[True, True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def binary_cross_entropy_with_logits_weighted_func(logits, targets, weight):
    logits_fp32 = logits.to(tl.float32)
    targets_fp32 = targets.to(tl.float32)
    weight_fp32 = weight.to(tl.float32)

    abs_logits = tl.abs(logits_fp32)
    max_term = tl.maximum(logits_fp32, 0.0)
    loss_fp32 = max_term - logits_fp32 * targets_fp32 + tl.log(1.0 + tl.exp(-abs_logits))

    final_loss_fp32 = loss_fp32 * weight_fp32
    loss = final_loss_fp32.to(logits.type.scalar)
    return loss


@pointwise_dynamic(is_tensor=[True, True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def binary_cross_entropy_with_logits_pos_weight_func(logits, targets, pos_weight):
    logits_fp32 = logits.to(tl.float32)
    targets_fp32 = targets.to(tl.float32)
    pos_weight_fp32 = pos_weight.to(tl.float32)

    abs_logits = tl.abs(logits_fp32)
    max_term = tl.maximum(logits_fp32, 0.0)
    log_term = tl.log(1.0 + tl.exp(-abs_logits))
    neg_part = (1.0 - targets_fp32) * (max_term + log_term)
    pos_part = pos_weight_fp32 * targets_fp32 * (max_term + log_term - logits_fp32)

    loss = (pos_part + neg_part).to(logits.type.scalar)
    return loss


@pointwise_dynamic(is_tensor=[True, True, True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def binary_cross_entropy_with_logits_full_func(logits, targets, weight, pos_weight):
    logits_fp32 = logits.to(tl.float32)
    targets_fp32 = targets.to(tl.float32)
    weight_fp32 = weight.to(tl.float32)
    pos_weight_fp32 = pos_weight.to(tl.float32)

    abs_logits = tl.abs(logits_fp32)
    max_term = tl.maximum(logits_fp32, 0.0)
    log_term = tl.log(1.0 + tl.exp(-abs_logits))
    neg_part = (1.0 - targets_fp32) * (max_term + log_term)
    pos_part = pos_weight_fp32 * targets_fp32 * (max_term + log_term - logits_fp32)
    weighted_loss = (pos_part + neg_part) * weight_fp32

    return weighted_loss.to(logits.type.scalar)


def binary_cross_entropy_with_logits(
    self, target, weight=None, pos_weight=None, reduction=1
):
    """
    Binary cross entropy with logits.

    Args:
        self: Input logits tensor
        target: Target labels tensor
        weight: Optional weight tensor
        pos_weight: Optional positive weight tensor
        reduction: 0=none, 1=mean, 2=sum

    Returns:
        Binary cross entropy loss
    """
    logger.debug("ILUVATAR GEMS binary_cross_entropy_with_logits")

    # Handle empty tensors
    if self.numel() == 0:
        result = torch.empty_like(self)
        if reduction == 1:
            result = result.mean()
        elif reduction == 2:
            result = result.sum()
        return result

    # Determine which kernel to use based on weight and pos_weight
    if weight is not None and pos_weight is not None:
        loss = binary_cross_entropy_with_logits_full_func(self, target, weight, pos_weight)
    elif weight is not None:
        loss = binary_cross_entropy_with_logits_weighted_func(self, target, weight)
    elif pos_weight is not None:
        loss = binary_cross_entropy_with_logits_pos_weight_func(self, target, pos_weight)
    else:
        loss = binary_cross_entropy_with_logits_func(self, target)

    # Apply reduction
    if reduction == 0:  # NONE
        return loss
    elif reduction == 1:  # MEAN
        return loss.mean()
    elif reduction == 2:  # SUM
        return loss.sum()
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")