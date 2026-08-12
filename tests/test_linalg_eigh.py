# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager

import flag_gems
import pytest
import torch

from . import accuracy_utils as utils


@contextmanager
def _ieee_float32_matmul():
    """Force IEEE float32 matmul (disable TF32) for the block, restore after.

    TF32 (10-bit mantissa) on Ampere+ GPUs introduces ~1e-3 errors in float32
    matmul, which corrupts the reconstruction verification of eigh
    (V @ diag(w) @ V.T). Disable it for the verification matmul only; the op
    under test is unaffected.
    """
    m = torch.backends.cuda.matmul
    use_new = hasattr(m, "fp32_precision")
    if use_new:
        old = m.fp32_precision
        m.fp32_precision = "ieee"
    else:
        old = m.allow_tf32
        m.allow_tf32 = False
    try:
        yield
    finally:
        if use_new:
            m.fp32_precision = old
        else:
            m.allow_tf32 = old


# linalg_eigh only supports float32/float64.
#
# The operator (src/flag_gems/ops/linalg_eigh.py) has three execution paths:
#   - n == 2  -> Triton `_eig_2x2_kernel` (analytical 2x2 formula, on device).
#   - n > 2   -> raises NotImplementedError.
#   - n < 2   -> host-side on device: diagonal as eigenvalues, identity as eigenvectors.
# Shapes below are split so each path is explicitly exercised and labelled.

# Path A: hits the Triton 2x2 kernel.
EIG_2X2_SHAPES = [(2, 2)]

# Path B: n>2 raises NotImplementedError.
EIG_NOTIMPL_SHAPES = [(3, 3), (5, 5), (8, 8), (16, 16), (32, 32)]

# Path C: n<2 (0x0 / 1x1), computed on device.
EIG_TRIVIAL_SHAPES = [(0, 0), (1, 1)]

# Batched variants: (2, 2, 2) hits the 2x2 kernel per batch element; the rest
# raise NotImplementedError.
EIG_BATCH_2X2_SHAPES = [(2, 2, 2)]
EIG_BATCH_NOTIMPL_SHAPES = [(4, 3, 3), (1, 8, 8)]
EIG_BATCH_TRIVIAL_SHAPES = [(2, 0, 0), (3, 1, 1)]


def make_symmetric_matrix(shape, dtype, device):
    """Create a symmetric matrix for eigendecomposition."""
    A = torch.randn(shape, dtype=dtype, device=device)
    A = (A + A.transpose(-2, -1)) / 2
    return A


def _assert_orthonormal(v, atol=1e-2):
    """Columns of v are eigenvectors: Vᵀ V ≈ I.

    Avoids comparing eigenvectors elementwise, since v and -v are both valid
    eigenvectors (sign ambiguity).
    """
    n = v.shape[-1]
    eye = torch.eye(n, dtype=v.dtype, device=v.device)
    gram = v.transpose(-2, -1) @ v
    expected = utils.to_reference(eye.expand_as(gram), False)
    utils.gems_assert_close(gram, expected, gram.dtype, atol=atol)


def _check_eigh_decomposition(A, eigenvalues, eigenvectors, atol=1e-3):
    """Verify the eigendecomposition via the defining relation A = V diag(w) Vᵀ.

    This is sign-ambiguous-free: any valid eigenbasis reconstructs A and is
    orthonormal, regardless of per-vector sign choices.
    """
    with _ieee_float32_matmul():
        reconstructed = (
            eigenvectors
            @ torch.diag_embed(eigenvalues).to(eigenvectors.dtype)
            @ eigenvectors.transpose(-2, -1)
        )
        ref_A = utils.to_reference(A, False)
        utils.gems_assert_close(reconstructed, ref_A, reconstructed.dtype, atol=atol)
        _assert_orthonormal(eigenvectors)


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_2X2_SHAPES,
    ids=[f"gpu_kernel_2x2-{s[0]}x{s[1]}" for s in EIG_2X2_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_2x2_kernel(shape, dtype):
    """n == 2: exercised by the Triton `_eig_2x2_kernel` on device."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp)

    # Eigenvalues
    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
    # Eigenvectors: reconstruct + orthonormality (sign-ambiguous-free)
    _check_eigh_decomposition(inp, res_out[0], res_out[1])


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_NOTIMPL_SHAPES,
    ids=[f"not_implemented_n{s[0]}" for s in EIG_NOTIMPL_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_not_implemented(shape, dtype):
    """n > 2: the op raises NotImplementedError."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    with flag_gems.use_gems():
        with pytest.raises(NotImplementedError):
            torch.linalg.eigh(inp)


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_BATCH_2X2_SHAPES,
    ids=[f"gpu_kernel_2x2-batch{s[0]}" for s in EIG_BATCH_2X2_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_batch_2x2_kernel(shape, dtype):
    """Batched n == 2: each batch element hits the Triton 2x2 kernel."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp)

    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
    _check_eigh_decomposition(inp, res_out[0], res_out[1])


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_BATCH_NOTIMPL_SHAPES,
    ids=[f"not_implemented-batch{s[-1]}" for s in EIG_BATCH_NOTIMPL_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_batch_not_implemented(shape, dtype):
    """Batched n > 2: the op raises NotImplementedError."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    with flag_gems.use_gems():
        with pytest.raises(NotImplementedError):
            torch.linalg.eigh(inp)


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_TRIVIAL_SHAPES,
    ids=[f"trivial_n{s[0]}" for s in EIG_TRIVIAL_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_trivial(shape, dtype):
    """n < 2 (0x0 / 1x1): computed on device via host-side ops."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp)

    # Eigenvalues
    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
    # Eigenvectors: reconstruct + orthonormality (sign-ambiguous-free)
    _check_eigh_decomposition(inp, res_out[0], res_out[1])


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_BATCH_TRIVIAL_SHAPES,
    ids=[f"trivial-batch_n{s[-1]}" for s in EIG_BATCH_TRIVIAL_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_batch_trivial(shape, dtype):
    """Batched n < 2 (0x0 / 1x1): computed on device via host-side ops."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp)

    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
    _check_eigh_decomposition(inp, res_out[0], res_out[1])
