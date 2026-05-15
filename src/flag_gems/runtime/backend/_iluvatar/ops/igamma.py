import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _igamma_series_kernel(a_ptr, x_ptr, result_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Compute lower incomplete gamma function: γ(a, x)
    Using series expansion: γ(a,x) = x^a * e^(-x) * Σ(n=0 to ∞) x^n / (a)(a+1)...(a+n)
    """
    pid = tl.program_id(0)
    num_elements = n_elements
    offset = pid * BLOCK_SIZE
    offs = offset + tl.arange(0, BLOCK_SIZE)

    mask = offs < num_elements
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Handle edge cases: a <= 0 or x <= 0 returns 0
    valid = (a > 0.0) & (x > 0.0)

    # Series expansion
    max_iter = 48

    # x^a * e^(-x)
    x_a_exp_neg_x = tl.exp(a * tl.log(x) - x)

    # Initial term: 1/a
    # This is the first term in the sum Σ(n=0) x^n / a = 1/a
    term = 1.0 / a

    sum_val = term
    a_curr = a

    for i in range(1, max_iter):
        a_curr = a_curr + 1.0
        term = term * x / a_curr
        sum_val = sum_val + term

    # γ(a,x) = x^a * e^(-x) * sum
    result = x_a_exp_neg_x * sum_val

    # Handle invalid cases
    result = tl.where(valid, result, 0.0)

    tl.store(result_ptr + offs, result, mask=mask)


def igamma(a, b):
    """Lower incomplete gamma function: γ(a, b)"""
    logger.debug("ILUVATAR GEMS IGAMMA")

    # Handle scalar inputs
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=torch.float32)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, dtype=torch.float32)

    # Convert to tensor on correct device
    if not a.is_cuda:
        a = a.to(b.device) if b.is_cuda else a
    if not b.is_cuda:
        b = b.to(a.device) if a.is_cuda else b

    # Handle empty tensors
    if a.numel() == 0 or b.numel() == 0:
        return torch.empty(0, device=a.device)

    # Get output shape (broadcast)
    output_shape = torch.broadcast_shapes(a.shape, b.shape)

    # Flatten inputs
    a_flat = a.expand(output_shape).contiguous().flatten()
    b_flat = b.expand(output_shape).contiguous().flatten()

    n_elements = a_flat.numel()

    # Allocate output
    result = torch.empty_like(a_flat)

    # Configure kernel
    BLOCK_SIZE = 1024
    num_warps = 4

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    with torch_device_fn.device(a_flat.device):
        _igamma_series_kernel[grid](
            a_flat, b_flat, result, n_elements,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
        )

    return result.reshape(output_shape)


def igamma_(a, b):
    """In-place version - not supported"""
    raise NotImplementedError("igamma_ in-place is not supported")