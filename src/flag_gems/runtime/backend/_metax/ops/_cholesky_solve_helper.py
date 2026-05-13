import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)


# The _cholesky_solve_helper function solves a linear system Ax = b
# where A is a positive definite matrix, given its Cholesky decomposition.
# If upper=False: A = L @ L.T, solve L @ L.T @ x = b
# If upper=True: A = U.T @ U, solve U.T @ U @ x = b
# Input A should already be the Cholesky factor (L or U).
# self is the right-hand side b, A is the Cholesky factor.


@libentry()
@triton.jit
def cholesky_solve_helper_kernel(
    b_ptr,
    L_ptr,
    output_ptr,
    N,
    K,
    stride_b_batch,
    stride_b_n,
    stride_b_k,
    stride_L_batch,
    stride_L_n,
    stride_L_n2,
    stride_out_batch,
    stride_out_n,
    stride_out_k,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel placeholder for Cholesky solve helper.
    Note: This is a placeholder - actual computation uses torch.cholesky_solve.
    """
    # Get program ID
    batch_idx = tle.program_id(0)
    col_idx = tle.program_id(1)

    if col_idx >= K:
        return


def cholesky_solve_helper(self_tensor, A_tensor, upper=False):
    """
    Metax specialized implementation of _cholesky_solve_helper.
    This is a wrapper that calls torch.cholesky_solve with proper logging.
    """
    logger.debug(
        "METAX GEMS CHOLESKY_SOLVE_HELPER: self.shape=%s, A.shape=%s, upper=%s",
        self_tensor.shape,
        A_tensor.shape,
        upper,
    )

    # Delegate to torch.cholesky_solve
    # Note: In a full implementation, this would use a custom Triton kernel
    return torch.cholesky_solve(self_tensor, A_tensor, upper=upper)