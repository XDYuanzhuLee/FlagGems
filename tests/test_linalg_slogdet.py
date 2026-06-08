import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Square matrix shapes: batched + various square sizes
SLOGDET_SHAPES = [(2, 3, 3), (4, 4), (8, 8), (16, 16), (32, 32)]


@pytest.mark.linalg_slogdet
@pytest.mark.parametrize("shape", SLOGDET_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_slogdet(shape, dtype):
    """Test linalg_slogdet accuracy against PyTorch reference."""
    assert len(shape) >= 2 and shape[-1] == shape[-2], "Input must be square matrix"

    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_A = utils.to_reference(A)

    ref_out = torch.linalg.slogdet(ref_A)

    with flag_gems.use_gems():
        res_out = torch.linalg.slogdet(A)

    utils.gems_assert_close(res_out.sign, ref_out.sign, dtype)
    utils.gems_assert_close(res_out.logabsdet, ref_out.logabsdet, dtype, atol=2e-3)


@pytest.mark.linalg_slogdet
def test_linalg_slogdet_zero_leading_pivot():
    """Permutation matrix with zero leading pivot — pivoting must handle this.

    A = [[0,1,0],[1,0,0],[0,0,1]] is non-singular (det=-1) but A[0,0]=0.
    With partial pivoting the kernel swaps row 0 with row 1 and computes the
    correct sign and logabsdet.
    """
    A = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=flag_gems.device,
    )

    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.slogdet(ref_A)

    with flag_gems.use_gems():
        res_out = torch.linalg.slogdet(A)

    utils.gems_assert_close(res_out.sign, ref_out.sign, torch.float32)
    utils.gems_assert_close(res_out.logabsdet, ref_out.logabsdet, torch.float32)


@pytest.mark.linalg_slogdet
def test_linalg_slogdet_negative_det():
    """Matrix with negative determinant: diag(1,-2,3) has det=-6, sign=-1."""
    A = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=torch.float32,
        device=flag_gems.device,
    )

    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.slogdet(ref_A)

    with flag_gems.use_gems():
        res_out = torch.linalg.slogdet(A)

    utils.gems_assert_close(res_out.sign, ref_out.sign, torch.float32)
    utils.gems_assert_close(res_out.logabsdet, ref_out.logabsdet, torch.float32)


@pytest.mark.linalg_slogdet
def test_linalg_slogdet_singular():
    """Rank-1 singular matrix: PyTorch returns sign=0, logabsdet=-inf."""
    A = torch.tensor(
        [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]],
        dtype=torch.float32,
        device=flag_gems.device,
    )

    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.slogdet(ref_A)

    with flag_gems.use_gems():
        res_out = torch.linalg.slogdet(A)

    utils.gems_assert_close(res_out.sign, ref_out.sign, torch.float32)
    utils.gems_assert_close(res_out.logabsdet, ref_out.logabsdet, torch.float32)


@pytest.mark.linalg_slogdet
def test_linalg_slogdet_batched():
    """Batched input with mixed well-conditioned and singular matrices."""
    # Batch of 3 matrices: well-conditioned, negative-det, singular
    A = torch.zeros(3, 3, 3, dtype=torch.float32, device=flag_gems.device)
    A[0] = torch.tensor(
        [[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]],
        dtype=torch.float32,
    )
    A[1] = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=torch.float32,
    )
    A[2] = torch.tensor(
        [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]],
        dtype=torch.float32,
    )

    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.slogdet(ref_A)

    with flag_gems.use_gems():
        res_out = torch.linalg.slogdet(A)

    utils.gems_assert_close(res_out.sign, ref_out.sign, torch.float32)
    utils.gems_assert_close(
        res_out.logabsdet, ref_out.logabsdet, torch.float32, atol=2e-3
    )


@pytest.mark.linalg_slogdet
def test_linalg_slogdet_aten_outputs():
    """Verify that the internal slogdet() returns valid LU and pivots tensors."""
    A = torch.randn(3, 3, dtype=torch.float32, device=flag_gems.device)

    with flag_gems.use_gems():
        # Call the internal slogdet directly to get all 4 outputs
        from flag_gems.ops.linalg_slogdet import slogdet as gems_slogdet

        sign, logabsdet, LU, pivots = gems_slogdet(A)

    # Check shapes and dtypes
    assert sign.shape == ()
    assert logabsdet.shape == ()
    assert LU.shape == A.shape
    assert pivots.shape == (3,)
    assert pivots.dtype == torch.int32

    # Pivot indices must be in [1, n] (1-based)
    assert (pivots >= 1).all()
    assert (pivots <= 3).all()

    # LU should be finite
    assert torch.isfinite(LU).all()
