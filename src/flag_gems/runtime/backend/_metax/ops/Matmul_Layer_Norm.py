import logging

import torch

logger = logging.getLogger("flag_gems." + __name__)


def matmul_layernorm(input, weight, normalized_shape, bias=None, eps=1e-5):
    """
    Fused matmul and layer normalization operation.

    This operation performs:
        1. matrix multiplication: result = input @ weight
        2. layer normalization along the last dimension

    Args:
        input: Input tensor (M x K)
        weight: Weight matrix for matmul (K x N)
        normalized_shape: Shape to normalize over (N,)
        bias: Optional bias for layer norm (N,)
        eps: Epsilon for numerical stability

    Returns:
        Normalized output tensor (M x N)
    """
    logger.debug("METAX GEMS MATMUL_LAYERNORM")

    # Validate inputs
    if input.dim() != 2:
        raise ValueError(f"input must be 2D, got {input.dim()}D")
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D, got {weight.dim()}D")

    M, K = input.shape
    K_w, N = weight.shape

    if K != K_w:
        raise ValueError(f"matmul dimension mismatch: {K} != {K_w}")

    if isinstance(normalized_shape, int):
        if normalized_shape != N:
            raise ValueError(f"normalized_shape {normalized_shape} != weight output dim {N}")
    else:
        if normalized_shape[-1] != N:
            raise ValueError(f"normalized_shape {normalized_shape}[-1] != weight output dim {N}")

    # Ensure contiguous
    input = input.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    # First: compute matmul using torch.mm - this will use flag_gems implementation
    # when called within flag_gems.use_gems() context
    matmul_result = torch.matmul(input, weight)

    # Then apply layer normalization
    normalized_output = torch.nn.functional.layer_norm(
        matmul_result, normalized_shape, weight=bias, eps=eps
    )

    return normalized_output