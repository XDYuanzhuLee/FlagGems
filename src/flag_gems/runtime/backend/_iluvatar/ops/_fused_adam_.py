import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _fused_adam_kernel(
    params_ptr,
    grads_ptr,
    exp_avgs_ptr,
    exp_avg_sqs_ptr,
    max_exp_avg_sqs_ptr,
    numel,
    lr,
    one_minus_beta1,
    one_minus_beta2,
    beta1,
    beta2,
    weight_decay,
    eps,
    bias_correction1,
    bias_correction2,
    amsgrad: tl.constexpr,
    maximize: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < numel

    # Load data
    param = tl.load(params_ptr + off, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(grads_ptr + off, mask=mask, other=0.0).to(tl.float32)
    exp_avg = tl.load(exp_avgs_ptr + off, mask=mask, other=0.0).to(tl.float32)
    exp_avg_sq = tl.load(exp_avg_sqs_ptr + off, mask=mask, other=0.0).to(tl.float32)

    if maximize:
        grad = -grad

    # Apply weight decay (Adam style - L2 regularization: add to gradient)
    if weight_decay != 0.0:
        grad = grad + weight_decay * param

    # Update biased first moment estimate
    exp_avg = exp_avg * beta1 + grad * one_minus_beta1

    # Update biased second raw moment estimate
    exp_avg_sq = exp_avg_sq * beta2 + (grad * grad) * one_minus_beta2

    # Compute bias-corrected estimates
    bias_corrected_exp_avg = exp_avg / bias_correction1
    bias_corrected_exp_avg_sq = exp_avg_sq / bias_correction2

    # Compute the denominator
    denom = tl.sqrt(bias_corrected_exp_avg_sq) + eps

    # Compute the update
    update = bias_corrected_exp_avg / denom

    # Update parameters
    param = param - lr * update

    # Store updated values
    tl.store(params_ptr + off, param, mask=mask)
    tl.store(grads_ptr + off, grad if not maximize else -grad, mask=mask)
    tl.store(exp_avgs_ptr + off, exp_avg, mask=mask)
    tl.store(exp_avg_sqs_ptr + off, exp_avg_sq, mask=mask)

    # Handle amsgrad
    if amsgrad:
        max_exp_avg_sq = tl.load(max_exp_avg_sqs_ptr + off, mask=mask, other=0.0).to(tl.float32)
        max_exp_avg_sq = tl.where(exp_avg_sq > max_exp_avg_sq, exp_avg_sq, max_exp_avg_sq)
        denom_amsgrad = tl.sqrt(max_exp_avg_sq / bias_correction2) + eps
        update_amsgrad = bias_corrected_exp_avg / denom_amsgrad
        param_amsgrad = param - lr * update_amsgrad
        tl.store(params_ptr + off, param_amsgrad, mask=mask)
        tl.store(max_exp_avg_sqs_ptr + off, max_exp_avg_sq, mask=mask)


def _fused_adam_(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    amsgrad=False,
    maximize=False,
    grad_scale=None,
    found_inf=None,
):
    logger.debug("ILUVATAR GEMS FUSED_ADAM")

    # Handle the case where inputs are tuples/lists of tensors
    if isinstance(params, (tuple, list)):
        params = params[0] if len(params) > 0 else params
    if isinstance(grads, (tuple, list)):
        grads = grads[0] if len(grads) > 0 else grads
    if isinstance(exp_avgs, (tuple, list)):
        exp_avgs = exp_avgs[0] if len(exp_avgs) > 0 else exp_avgs
    if isinstance(exp_avg_sqs, (tuple, list)):
        exp_avg_sqs = exp_avg_sqs[0] if len(exp_avg_sqs) > 0 else exp_avg_sqs
    if isinstance(max_exp_avg_sqs, (tuple, list)):
        max_exp_avg_sqs = max_exp_avg_sqs[0] if len(max_exp_avg_sqs) > 0 else max_exp_avg_sqs
    if isinstance(state_steps, (tuple, list)):
        state_steps = state_steps[0] if len(state_steps) > 0 else state_steps

    # Handle None max_exp_avg_sqs for non-amsgrad
    if max_exp_avg_sqs is None:
        max_exp_avg_sqs = torch.zeros_like(exp_avg_sqs)

    # Compute bias correction factors from state_steps
    if state_steps is not None and state_steps.numel() > 0:
        max_step = state_steps.max().item()
    else:
        max_step = 0

    bias_correction1 = 1.0 - beta1 ** (max_step + 1)
    bias_correction2 = 1.0 - beta2 ** (max_step + 1)

    if bias_correction1 == 0:
        bias_correction1 = 1.0
    if bias_correction2 == 0:
        bias_correction2 = 1.0

    # Compute one_minus_beta values
    one_minus_beta1 = 1.0 - beta1
    one_minus_beta2 = 1.0 - beta2

    # Get numel
    numel = params.numel()

    # Define block size
    BLOCK_SIZE = 2048

    # Compute grid
    grid = ((numel + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch kernel
    with torch_device_fn.device(params.device):
        _fused_adam_kernel[grid](
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            numel,
            lr,
            one_minus_beta1,
            one_minus_beta2,
            beta1,
            beta2,
            weight_decay,
            eps,
            bias_correction1,
            bias_correction2,
            amsgrad,
            maximize,
            BLOCK_SIZE,
        )

    return None