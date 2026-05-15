import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger("flag_gems." + __name__)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def beam_search_score_kernel(a, b):
    return a + b


def beam_search_score(A: torch.Tensor, B: torch.Tensor):
    """Beam search scoring function.

    This function computes the beam search score by adding two score tensors.
    In beam search decoding, this is used to accumulate log probabilities:
    - A: previous beam scores (log probabilities)
    - B: next token log probabilities
    - Output: updated beam scores (A + B)

    Args:
        A: Previous beam scores tensor
        B: Next token log probabilities tensor

    Returns:
        Updated beam scores (element-wise sum)
    """
    logger.debug("METAX GEMS BEAM_SEARCH_SCORE")
    # Let pointwise_dynamic handle broadcasting
    return beam_search_score_kernel(A, B)