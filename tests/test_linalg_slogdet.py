import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Square matrix shapes for slogdet: batch+square, and various square sizes
SLOGDET_SHAPES = [(2, 3, 3), (4, 4), (8, 8), (16, 16), (32, 32)]


@pytest.mark.linalg_slogdet
@pytest.mark.parametrize("shape", SLOGDET_SHAPES)
# linalg.slogdet only supports float32/float64 on CUDA; half-precision not supported
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_slogdet(shape, dtype):
    """Test linalg_slogdet accuracy against PyTorch reference."""
    # Ensure we have a square matrix
    assert len(shape) >= 2 and shape[-1] == shape[-2], "Input must be square matrix"

    # Create input tensor
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_A = utils.to_reference(A)

    # Compute reference
    ref_out = torch.linalg.slogdet(ref_A)

    # Compute with FlagGems
    with flag_gems.use_gems():
        res_out = torch.linalg.slogdet(A)

    # Compare sign
    utils.gems_assert_close(res_out.sign, ref_out.sign, dtype)

    # Compare logabsdet with relaxed tolerance: Gaussian elimination without partial
    # pivoting accumulates floating point error, especially for larger matrices.
    # Uses the same atol as test_svd.py for consistency with other matrix decompositions.
    utils.gems_assert_close(res_out.logabsdet, ref_out.logabsdet, dtype, atol=2e-3)


@pytest.mark.linalg_slogdet
@pytest.mark.xfail(
    reason=(
        "Gaussian elimination without pivoting cannot handle matrices with "
        "zero leading pivot. A permutation matrix that swaps the first row "
        "has A[0,0]=0, which the kernel treats as singular even though the "
        "matrix is non-singular (det=-1)."
    ),
    strict=True,
)
def test_linalg_slogdet_zero_leading_pivot():
    """Targeted test: permutation matrix with zero leading pivot.

    This matrix is non-singular (det=-1) but the current Gaussian elimination
    kernel without pivoting reads A[0,0]=0 as the first pivot and skips
    elimination, incorrectly reporting sign=0. PyTorch's LU decomposition with
    partial pivoting handles this correctly.
    """
    # 3x3 permutation matrix that swaps rows 0 and 1
    A = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=flag_gems.device,
    )

    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.slogdet(ref_A)

    with flag_gems.use_gems():
        res_out = torch.linalg.slogdet(A)

    # PyTorch reference: sign=-1, logabsdet=0 for this permutation matrix
    utils.gems_assert_close(res_out.sign, ref_out.sign, torch.float32)
    utils.gems_assert_close(res_out.logabsdet, ref_out.logabsdet, torch.float32)
