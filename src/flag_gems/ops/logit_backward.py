# Generated for FlagGems
import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit
def logit_backward_kernel(
    grad_output_ptr,
    self_ptr,
    output_ptr,
    n_elements,
    eps,
    HAS_EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_output = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)
    self_val = tl.load(self_ptr + offsets, mask=mask, other=0.0)

    if HAS_EPS:
        # PyTorch returns 0 when input is outside [eps, 1-eps]
        # This matches PyTorch's behavior
        x = tl.minimum(tl.maximum(self_val, eps), 1.0 - eps)
        # Only compute gradient if input was already in valid range
        in_range = (self_val >= eps) & (self_val <= 1.0 - eps)
        result = tl.where(in_range, grad_output / (x * (1.0 - x)), 0.0)
    else:
        result = grad_output / (self_val * (1.0 - self_val))

    tl.store(output_ptr + offsets, result, mask=mask)


def _logit_backward_impl(grad_output, self, eps=None):
    n_elements = grad_output.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    eps_val = float(eps) if eps is not None else 0.0
    has_eps = eps is not None

    output = torch.empty_like(grad_output)

    with torch_device_fn.device(grad_output.device):
        logit_backward_kernel[grid](
            grad_output,
            self,
            output,
            n_elements,
            eps_val,
            HAS_EPS=has_eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return output


def logit_backward(grad_output, self, eps=None):
    return _logit_backward_impl(grad_output, self, eps)