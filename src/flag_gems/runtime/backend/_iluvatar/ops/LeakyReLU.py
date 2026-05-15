import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def leaky_relu_kernel(x, negative_slope):
    zero = tl.zeros([1], dtype=x.dtype)
    slope = tl.full([1], negative_slope, dtype=x.dtype)
    # leaky_relu(x) = max(0, x) + negative_slope * min(0, x)
    # Equivalent, branchless:
    return tl.maximum(x, zero) + slope * tl.minimum(x, zero)


def leaky_relu(input: torch.Tensor, negative_slope: float = 0.01):
    """
    Iluvatar specialized LeakyReLU operator.
    ATen: ('leaky_relu', <Autograd.disable: False>)
    """
    logger.debug("ILUVATAR GEMS LEAKY_RELU")
    if input.numel() == 0:
        return torch.empty_like(input)
    return leaky_relu_kernel(input, negative_slope)


def leaky_relu_out(
    input: torch.Tensor, negative_slope: float = 0.01, out: torch.Tensor = None
):
    """
    Iluvatar specialized LeakyReLU operator with out parameter.
    ATen: ('leaky_relu.out', <Autograd.disable: False>)
    """
    logger.debug("ILUVATAR GEMS LEAKY_RELU_OUT")
    if out is None:
        raise ValueError("Argument 'out' must be provided for leaky_relu_out.")
    if input.numel() == 0:
        return out
    return leaky_relu_kernel(input, negative_slope, out0=out)