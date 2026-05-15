import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)


def linear_backward(
    self: torch.Tensor,
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    output_mask: tuple,
):
    """
    Compute the backward pass for torch.nn.functional.linear.

    Args:
        self: Input tensor (batch_size, input_dim)
        grad_output: Gradient of loss w.r.t. output (batch_size, output_dim)
        weight: Linear weight (output_dim, input_dim)
        output_mask: Tuple of 3 booleans (grad_input, grad_weight, grad_bias)

    Returns:
        Tuple of (grad_input, grad_weight, grad_bias) based on output_mask
    """
    logger.debug("METAX GEMS LINEAR_BACKWARD")

    batch_size, input_dim = self.shape
    output_dim = weight.shape[0]

    grad_input = None
    grad_weight = None
    grad_bias = None

    # Ensure contiguity for efficient computation
    self = self.contiguous()
    grad_output = grad_output.contiguous()
    weight = weight.contiguous()

    # Compute grad_input = grad_output @ weight
    # grad_output: (M, N), weight: (N, K) => grad_input: (M, K)
    if output_mask[0]:
        grad_input = torch.mm(grad_output, weight)

    # Compute grad_weight = grad_output.T @ self
    # grad_output: (M, N), self: (M, K) => grad_weight: (N, K)
    if output_mask[1]:
        grad_weight = torch.mm(grad_output.t(), self)

    # Compute grad_bias = grad_output.sum(dim=0)
    # grad_output: (M, N) => grad_bias: (N,)
    if output_mask[2]:
        grad_bias = grad_output.sum(dim=0)

    return grad_input, grad_weight, grad_bias