import logging
import math

import torch
import triton

from flag_gems import runtime
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils import tl_extra_shim
from flag_gems import dropout as gems_dropout
from flag_gems import layer_norm as gems_layer_norm

logger = logging.getLogger("flag_gems." + __name__)


class LayerNormDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, weight=None, bias=None, eps=1e-5, p=0.5, train=True):
        logger.debug("METAX GEMS LAYERNORM_DROPOUT FORWARD")

        # Apply LayerNorm first
        normalized, mean, rstd = gems_layer_norm(
            input,
            normalized_shape,
            weight=weight,
            bias=bias,
            eps=eps
        )

        # Apply Dropout during training
        if train and p > 0 and p < 1:
            normalized, mask = gems_dropout(normalized, p=p, train=True)

        if input.requires_grad:
            ctx.save_for_backward(input, weight, bias, mean, rstd)
            ctx.normalized_shape = normalized_shape
            ctx.p = p
            ctx.train = train
            # If dropout was applied, we need to handle it in backward
            if train and p > 0 and p < 1:
                ctx.mask = mask
            else:
                ctx.mask = None

        if train and p > 0 and p < 1:
            return normalized
        return normalized

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("METAX GEMS LAYERNORM_DROPOUT BACKWARD")

        input, weight, bias, mean, rstd = ctx.saved_tensors
        normalized_shape = ctx.normalized_shape
        p = ctx.p
        train = ctx.train
        mask = ctx.mask

        N = math.prod(normalized_shape)
        M = input.numel() // N

        # Scale grad by dropout mask during backward if dropout was applied
        if train and p > 0 and p < 1 and mask is not None:
            scale = 1.0 / (1.0 - p)
            grad_output = grad_output * mask * scale

        # Compute LayerNorm backward
        output_mask = [True, weight is not None, bias is not None]
        grad_input, grad_weight, grad_bias = gems_layer_norm.layer_norm_backward(
            grad_output,
            input,
            normalized_shape,
            mean,
            rstd,
            weight=weight,
            bias=bias,
            output_mask=output_mask
        )

        return grad_input, None, grad_weight, grad_bias, None, None, None


def layer_norm_dropout(input, normalized_shape, weight=None, bias=None, eps=1e-5, p=0.5, train=True):
    """
    Fused LayerNorm + Dropout operator.

    This implementation uses the existing LayerNorm and Dropout operations
    from FlagGems. For the Metax backend, this provides compatibility
    while still leveraging the optimized implementations.

    Args:
        input: Input tensor
        normalized_shape: Shape to normalize over (last dimensions)
        weight: Optional weight tensor for LayerNorm
        bias: Optional bias tensor for LayerNorm
        eps: Epsilon for numerical stability in LayerNorm
        p: Dropout probability
        train: Whether to apply dropout (training mode)

    Returns:
        Output tensor after LayerNorm and Dropout
    """
    return LayerNormDropout.apply(input, normalized_shape, weight, bias, eps, p, train)