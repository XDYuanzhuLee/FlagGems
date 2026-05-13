import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)


@libentry()
@triton.jit(do_not_specialize=["lr", "beta1", "beta2", "weight_decay", "eps"])
def _fused_adam_kernel(
    self_ptr,
    grads_ptr,
    exp_avgs_ptr,
    exp_avg_sqs_ptr,
    max_exp_avg_sqs_ptr,
    n_elements,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    amsgrad: tl.constexpr,
    maximize: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load parameters
    self = tl.load(self_ptr + offsets, mask=mask, other=0.0)
    grad = tl.load(grads_ptr + offsets, mask=mask, other=0.0)
    exp_avg = tl.load(exp_avgs_ptr + offsets, mask=mask, other=0.0)
    exp_avg_sq = tl.load(exp_avg_sqs_ptr + offsets, mask=mask, other=0.0)

    # Apply maximize (negate gradient if maximizing)
    if maximize:
        grad = -grad

    # Cast scalar parameters to match tensor dtype for type consistency
    # Use the tensor's dtype to determine the type
    dtype = self.dtype
    one_minus_beta1 = 1.0 - beta1
    one_minus_beta2 = 1.0 - beta2

    # Update biased first moment estimate: exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    exp_avg = exp_avg * beta1 + grad * one_minus_beta1

    # Update biased second raw moment estimate: exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
    exp_avg_sq = exp_avg_sq * beta2 + (grad * grad) * one_minus_beta2

    # Compute the denominator
    denom = tl.sqrt(exp_avg_sq) + eps

    # Compute the update step
    update = exp_avg / denom

    # Apply weight decay (L2 regularization): self = self - weight_decay * lr * self
    # Using the formula: self = self * (1 - weight_decay * lr)
    decay_factor = 1.0 - weight_decay * lr
    self = self * decay_factor

    # Apply the update: self = self - lr * update
    self = self - lr * update

    # Store the updated values
    tl.store(self_ptr + offsets, self, mask=mask)
    tl.store(grads_ptr + offsets, grad, mask=mask)
    tl.store(exp_avgs_ptr + offsets, exp_avg, mask=mask)
    tl.store(exp_avg_sqs_ptr + offsets, exp_avg_sq, mask=mask)

    # Handle AMSGrad if enabled
    if amsgrad:
        max_exp_avg_sq = tl.load(max_exp_avg_sqs_ptr + offsets, mask=mask, other=0.0)
        max_exp_avg_sq = tl.maximum(max_exp_avg_sq, exp_avg_sq)
        tl.store(max_exp_avg_sqs_ptr + offsets, max_exp_avg_sq, mask=mask)


def _fused_adam_(
    self,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.0,
    eps=1e-8,
    amsgrad=False,
    maximize=False,
    grad_scale=None,
    found_inf=None,
):
    logger.debug("METAX GEMS FUSED_ADAM_")

    # Handle the case where tensors are passed as lists
    if isinstance(self, (list, tuple)):
        num_params = len(self)
    else:
        num_params = 1
        self = [self]
        grads = [grads]
        exp_avgs = [exp_avgs]
        exp_avg_sqs = [exp_avg_sqs]
        max_exp_avg_sqs = [max_exp_avg_sqs] if amsgrad else [None]
        state_steps = [state_steps]

    # Process each parameter tensor
    for i in range(num_params):
        param = self[i]
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        max_exp_avg_sq = max_exp_avg_sqs[i] if amsgrad else None

        # Ensure tensors are contiguous
        param = param.contiguous()
        grad = grad.contiguous()
        exp_avg = exp_avg.contiguous()
        exp_avg_sq = exp_avg_sq.contiguous()

        n_elements = param.numel()

        # Determine block size based on tensor size
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        with torch_device_fn.device(param.device):
            if amsgrad and max_exp_avg_sq is not None:
                max_exp_avg_sq = max_exp_avg_sq.contiguous()
                _fused_adam_kernel[grid](
                    param,
                    grad,
                    exp_avg,
                    exp_avg_sq,
                    max_exp_avg_sq,
                    n_elements,
                    lr,
                    beta1,
                    beta2,
                    weight_decay,
                    eps,
                    amsgrad,
                    maximize,
                    BLOCK_SIZE=BLOCK_SIZE,
                )
            else:
                # Dummy pointer for max_exp_avg_sqs when amsgrad is False
                _fused_adam_kernel[grid](
                    param,
                    grad,
                    exp_avg,
                    exp_avg_sq,
                    param,  # Use param as dummy pointer
                    n_elements,
                    lr,
                    beta1,
                    beta2,
                    weight_decay,
                    eps,
                    amsgrad,
                    maximize,
                    BLOCK_SIZE=BLOCK_SIZE,
                )

    return None