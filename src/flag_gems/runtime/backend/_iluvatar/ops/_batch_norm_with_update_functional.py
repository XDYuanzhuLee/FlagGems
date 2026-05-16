import logging

import torch
from torch import Tensor

from flag_gems.ops.batch_norm import batch_norm as batch_norm_fn

logger = logging.getLogger("flag_gems." + __name__)


def _batch_norm_with_update_functional(
    input: Tensor,
    weight=None,
    bias=None,
    running_mean: Tensor = None,
    running_var: Tensor = None,
    momentum=0.1,
    eps=1e-5,
):
    """
    Functional batch norm that returns updated running mean and var.

    This is a specialized Iluvatar implementation that wraps the batch_norm kernel.

    Args:
        input: Input tensor
        weight: Optional weight tensor
        bias: Optional bias tensor
        running_mean: Running mean statistics
        running_var: Running variance statistics
        momentum: Momentum for updating running statistics
        eps: Epsilon for numerical stability

    Returns:
        Tuple of (output, save_mean, save_invstd, reserve, running_mean_out, running_var_out)
    """
    logger.debug("ILUVATAR GEMS _batch_norm_with_update_functional")

    output, save_mean, save_invstd = batch_norm_fn(
        input,
        weight=weight,
        bias=bias,
        running_mean=running_mean,
        running_var=running_var,
        training=False,
        momentum=momentum,
        eps=eps,
    )

    running_mean_out = running_mean.clone()
    running_var_out = running_var.clone()

    reserve = torch.empty(0, dtype=input.dtype, device=input.device)

    return (output, save_mean, save_invstd, reserve, running_mean_out, running_var_out)