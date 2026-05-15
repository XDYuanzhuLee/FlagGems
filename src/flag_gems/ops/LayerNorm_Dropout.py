import logging

import torch

logger = logging.getLogger("flag_gems." + __name__)


def layer_norm_dropout(input, normalized_shape, weight=None, bias=None, eps=1e-5, p=0.5, train=True):
    """
    Fused LayerNorm + Dropout operator.

    This is a reference implementation that calls torch.layer_norm followed by
    torch.nn.functional.dropout. The Metax backend provides an optimized Triton
    kernel implementation.

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
    logger.debug("GEMS LAYERNORM_DROPOUT FORWARD")

    # Apply LayerNorm
    normalized = torch.nn.functional.layer_norm(
        input,
        normalized_shape,
        weight=weight,
        bias=bias,
        eps=eps
    )

    # Apply Dropout during training
    if train and p > 0 and p < 1:
        normalized = torch.nn.functional.dropout(normalized, p=p, training=True)

    return normalized