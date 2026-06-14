import logging

import torch

from flag_gems.ops.special_chebyshev_polynomial_u import special_chebyshev_polynomial_u_kernel

logger = logging.getLogger("flag_gems." + __name__)


def special_chebyshev_polynomial_u(x, n):
    """Compute the Chebyshev polynomial of the second kind U_n(x).

    Args:
        x: The input tensor.
        n: The order of the polynomial. Can be a scalar, int, float, or tensor.

    Returns:
        The Chebyshev polynomial U_n(x) evaluated at x.
    """
    logger.debug("GEMS_METAX SPECIAL_CHEBYSHEV_POLYNOMIAL_U")
    # torch.special.chebyshev_polynomial_u only supports float32 on CUDA
    assert x.dtype == torch.float32, f"unsupported dtype {x.dtype}"

    # Convert n to tensor if it's a scalar
    if not isinstance(n, torch.Tensor):
        n = torch.tensor(n, dtype=torch.int32, device=x.device)

    # Ensure n has the same shape as x for broadcasting
    if n.shape != x.shape:
        n = n.expand(x.shape)

    return special_chebyshev_polynomial_u_kernel(x, n)
