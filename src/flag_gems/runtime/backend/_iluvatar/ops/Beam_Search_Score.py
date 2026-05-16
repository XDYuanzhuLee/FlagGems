import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def beam_search_score_func(x, y):
    return x + y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def beam_search_score_func_tensor_scalar(x, y):
    return x + y


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def beam_search_score_func_scalar_tensor(x, y):
    return x + y


def beam_search_score(A, B):
    """Iluvatar specialized implementation of beam search score computation.

    Compute beam search score by adding accumulated beam scores with next token scores.
    This is a simple addition operation used in beam search to accumulate scores.

    Args:
        A: Previous beam scores (tensor or scalar)
        B: Next token scores/log probabilities (tensor or scalar)

    Returns:
        Updated beam scores
    """
    logger.debug("ILUVATAR GEMS BEAM_SEARCH_SCORE")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        return beam_search_score_func(A, B)
    elif isinstance(A, torch.Tensor):
        return beam_search_score_func_tensor_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return beam_search_score_func_scalar_tensor(A, B)
    else:
        return torch.tensor(A + B)