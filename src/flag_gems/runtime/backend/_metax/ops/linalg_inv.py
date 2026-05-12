import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)

# Unsupported dtypes for linalg.solve on this backend
UNSUPPORTED_DTYPES = {torch.float16, torch.bfloat16}


def linalg_inv(A):
    """
    Compute the inverse of a square matrix.

    This implementation uses torch.linalg.solve to compute the inverse by
    solving A @ X = I for X, where I is the identity matrix.
    For unsupported dtypes (float16, bfloat16), it uses a CPU fallback.
    """
    logger.debug("METAX GEMS LINALG_INV")
    # Ensure contiguous input
    A = A.contiguous()

    # Get the size of the matrix
    n = A.shape[-1]

    # Check if dtype is supported
    if A.dtype in UNSUPPORTED_DTYPES:
        # Use CPU fallback for unsupported dtypes
        # Convert to float32 for computation, then convert back
        A_fp32 = A.to(torch.float32)
        A_fp32_cpu = A_fp32.cpu()
        result_cpu = torch.linalg.inv(A_fp32_cpu)
        result = result_cpu.to(A.device)
        return result.to(A.dtype)

    # Create identity matrix
    # Handle both 2D and batched inputs
    if A.ndim == 2:
        I = torch.eye(n, dtype=A.dtype, device=A.device)
    else:
        # Batched case: (*, n, n)
        batch_shape = A.shape[:-2]
        I = torch.eye(n, dtype=A.dtype, device=A.device)
        I = I.expand(*batch_shape, n, n)

    # Solve A @ X = I for X
    # This computes the inverse without calling linalg_inv
    return torch.linalg.solve(A, I)