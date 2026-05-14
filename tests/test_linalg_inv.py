import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Matrix inversion test shapes - square matrices (only 2x2 and 3x3 supported by Triton kernel)
MATRIX_SHAPES = [(2, 2), (3, 3)]
# Note: bfloat16 not supported by torch.linalg.inv directly, so we only test float32 and float16
MATRIX_DTYPES = [torch.float32, torch.float16]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.linalg_inv
@pytest.mark.parametrize("shape", MATRIX_SHAPES)
@pytest.mark.parametrize("dtype", MATRIX_DTYPES)
def test_accuracy_linalg_inv(shape, dtype):
    """Test linalg_inv accuracy against torch.linalg.inv"""
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Create invertible matrix
    n = shape[0]
    # Use random matrix and add identity to ensure it's invertible
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    A = A + torch.eye(n, dtype=dtype, device=flag_gems.device) * n

    # Convert to float32 for reference (torch.linalg.inv doesn't support bfloat16)
    ref_A = to_reference(A).to(torch.float32)

    ref_out = torch.linalg.inv(ref_A).to(dtype)
    with flag_gems.use_gems():
        res_out = torch.linalg.inv(A)

    # For matrix inverse, use relaxed tolerance for float16/bfloat16
    if dtype == torch.float32:
        gems_assert_close(res_out, ref_out, dtype, atol=1e-4)
    elif dtype == torch.float16:
        gems_assert_close(res_out, ref_out, dtype, atol=1e-2)
    else:  # bfloat16
        gems_assert_close(res_out, ref_out, dtype, atol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.linalg_inv
@pytest.mark.parametrize("shape", MATRIX_SHAPES)
@pytest.mark.parametrize("dtype", MATRIX_DTYPES)
@pytest.mark.skip(reason="Batched matrix inverse not yet supported")
def test_accuracy_linalg_inv_batched(shape, dtype):
    """Test linalg_inv accuracy for batched matrices"""
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Create batch of invertible matrices
    batch_size = 4
    n = shape[0]
    A = torch.randn(batch_size, n, n, dtype=dtype, device=flag_gems.device)
    # Add identity to ensure invertibility
    A = A + torch.eye(n, dtype=dtype, device=flag_gems.device).unsqueeze(0) * n

    # Convert to float32 for reference (torch.linalg.inv doesn't support bfloat16)
    ref_A = to_reference(A).to(torch.float32)

    ref_out = torch.linalg.inv(ref_A).to(dtype)
    with flag_gems.use_gems():
        res_out = torch.linalg.inv(A)

    # For matrix inverse, use relaxed tolerance for float16
    if dtype == torch.float32:
        gems_assert_close(res_out, ref_out, dtype, atol=1e-4)
    else:  # float16
        gems_assert_close(res_out, ref_out, dtype, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.inplace
@pytest.mark.linalg_inv_
@pytest.mark.parametrize("shape", MATRIX_SHAPES)
@pytest.mark.parametrize("dtype", MATRIX_DTYPES)
def test_accuracy_linalg_inv_(shape, dtype):
    """Test linalg_inv_ (in-place) accuracy"""
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    n = shape[0]
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    A = A + torch.eye(n, dtype=dtype, device=flag_gems.device) * n

    # Convert to float32 for reference (torch.linalg.inv doesn't support bfloat16)
    ref_A = to_reference(A.clone()).to(torch.float32)

    ref_out = ref_A.linalg_inv_().to(dtype)
    with flag_gems.use_gems():
        res_out = A.linalg_inv_()

    # For matrix inverse, use relaxed tolerance for float16
    if dtype == torch.float32:
        gems_assert_close(res_out, ref_out, dtype, atol=1e-4)
    else:  # float16
        gems_assert_close(res_out, ref_out, dtype, atol=1e-2)
