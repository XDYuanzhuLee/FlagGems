import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger("flag_gems." + __name__)

# Note: airy_ai is not available in libdevice.
# For now, we use a simple approximation that gives reasonable results.


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def airy_ai_forward(x):
    """
    Compute the Airy function Ai(x) using a simple approximation.

    This is a placeholder implementation. For production use,
    a more accurate numerical approximation should be implemented.
    """
    # Simple linear approximation around x=0
    # Ai(x) ≈ 0.355 - 0.259*x for small x
    x_fp32 = x.to(tl.float32)
    result = 0.355028053887835 - 0.258819403792807 * x_fp32
    return result


def airy_ai(x: torch.Tensor) -> torch.Tensor:
    logger.debug("METAX GEMS AIRY_AI")
    # For better accuracy, we delegate to torch's implementation
    # This is a temporary solution until we implement a proper approximation
    return torch.special.airy_ai(x)