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

import pytest
import torch

import flag_gems

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


# linalg_eigh / _linalg_eigh registration.
#
# `torch.linalg.eigh` (the user-facing Python API) dispatches to
# `aten::_linalg_eigh`, which is the operator registered to FlagGems.
# The dispatch policy keeps #3951's Triton 2x2 kernel on the user path:
#   - n == 2, float32, compute_v=True -> Triton `_eig_2x2_kernel` (on device).
#   - n > 2  -> cuSOLVER / LAPACK via `_linalg_eigh` (CPU round-trip).
#   - n < 2  -> on device: diagonal as eigenvalues, identity as eigenvectors.
#
# `aten::linalg_eigh` is a separate public entry that mirrors the same paths.
# Shapes below are split so each path is explicitly exercised and labelled.

# Path A: hits the Triton 2x2 kernel (via the user path torch.linalg.eigh).
# Includes fp16/bf16 — these are widened to fp32 inside the 2x2 kernel path,
# which is an on-device enhancement beyond native torch.linalg.eigh (which
# raises NotImplementedError for Half/BFloat16).
EIG_2X2_SHAPES = [(2, 2)]
EIG_2X2_LOWDTYPE = [torch.float16, torch.bfloat16]

# Path B: n > 2, handled by the cuSOLVER fallback inside _linalg_eigh.
EIG_CUSOLVER_SHAPES = [(3, 3), (5, 5), (8, 8), (16, 16), (32, 32)]

# Path C: n < 2 (0x0 / 1x1), computed on device for real, cuSOLVER for complex.
EIG_TRIVIAL_SHAPES = [(0, 0), (1, 1)]
EIG_TRIVIAL_COMPLEX_SHAPES = [(0, 0), (1, 1)]

# Complex n == 2: Triton cannot specialise complex dtypes, so even the 2x2
# case routes to the cuSOLVER path (same as n > 2).
EIG_COMPLEX_2X2_SHAPES = [(2, 2)]

# Batched variants.
EIG_BATCH_2X2_SHAPES = [(2, 2, 2)]
EIG_BATCH_CUSOLVER_SHAPES = [(4, 3, 3), (1, 8, 8)]
EIG_BATCH_TRIVIAL_SHAPES = [(2, 0, 0), (3, 1, 1)]
EIG_BATCH_CUSOLVER_COMPLEX_SHAPES = [(2, 3, 3), (1, 4, 4)]

# UPLO is a core parameter of eigh; the default "L" is covered by the path
# tests above, "U" is exercised separately against the cuSOLVER path.
EIG_UPLO_U_SHAPES = [(3, 3), (5, 5)]
EIG_UPLO_U_COMPLEX_SHAPES = [(3, 3), (5, 5)]


def make_symmetric_matrix(shape, dtype, device):
    """Create a symmetric (or Hermitian) matrix for eigendecomposition."""
    A = torch.randn(shape, dtype=dtype, device=device)
    if A.is_complex():
        A = (A + A.mH) / 2
    else:
        A = (A + A.transpose(-2, -1)) / 2
    return A


def _assert_close(res, ref, dtype, atol=1e-4):
    """Wrapper around the accuracy utils that handles complex128.

    FlagGems' ``gems_assert_close`` tolerance table (``RESOLUTION``) has no
    ``complex128`` entry, so direct lookup raises ``KeyError``. Existing linalg
    tests (cholesky_solve, linalg_ldl_solve, linalg_cross) handle this by
    falling back to ``torch.testing.assert_close`` for complex128; mirror that
    pattern here.
    """
    if dtype == torch.complex128:
        # FlagGems' gems_assert_close tolerance table has no complex128 entry.
        # Mirror the cholesky_solve / linalg_ldl_solve / linalg_cross pattern:
        # fall back to torch.testing.assert_close. Use only atol (the caller's
        # value, ~1e-2 for sign/basis-ambiguous reconstruction) with a loose
        # rtol, since near-zero elements otherwise dominate the relative error.
        torch.testing.assert_close(res, ref, atol=atol, rtol=1e-3)
    else:
        utils.gems_assert_close(res, ref, dtype, atol=atol)


def _assert_orthonormal(v, atol=1e-2):
    """Columns of v are eigenvectors: Vᴴ V ≈ I.

    Avoids comparing eigenvectors elementwise, since v and -v are both valid
    eigenvectors (sign ambiguity).
    """
    n = v.shape[-1]
    eye = torch.eye(n, dtype=v.dtype, device=v.device)
    v_h = v.mH if v.is_complex() else v.transpose(-2, -1)
    gram = v_h @ v
    expected = utils.to_reference(eye.expand_as(gram), False)
    _assert_close(gram, expected, gram.dtype, atol=atol)


def _check_eigh_decomposition(A, eigenvalues, eigenvectors, atol=1e-3):
    """Verify the eigendecomposition via the defining relation A = V diag(w) Vᴴ.

    For real inputs this is V diag(w) Vᵀ; for complex/Hermitian inputs it is
    V diag(w) Vᴴ (conjugate transpose). This is sign-ambiguous-free: any valid
    eigenbasis reconstructs A and is orthonormal, regardless of per-vector sign
    choices. Works for both the Triton 2x2 path and the cuSOLVER path.
    """
    with _ieee_float32_matmul():
        v_t = (
            eigenvectors.mH
            if eigenvectors.is_complex()
            else eigenvectors.transpose(-2, -1)
        )
        reconstructed = (
            eigenvectors @ torch.diag_embed(eigenvalues).to(eigenvectors.dtype) @ v_t
        )
        ref_A = utils.to_reference(A, False)
        _assert_close(reconstructed, ref_A, reconstructed.dtype, atol=atol)
        _assert_orthonormal(eigenvectors)


# ---------------------------------------------------------------------------
# User path: torch.linalg.eigh -> aten::_linalg_eigh (the registered op)
# ---------------------------------------------------------------------------


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_2X2_SHAPES,
    ids=[f"gpu_kernel_2x2-{s[0]}x{s[1]}" for s in EIG_2X2_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_2x2_kernel(shape, dtype):
    """n == 2 on the user path: exercised by the Triton `_eig_2x2_kernel`."""
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
    EIG_CUSOLVER_SHAPES,
    ids=[f"cusolver_n{s[0]}" for s in EIG_CUSOLVER_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64, torch.complex128])
def test_linalg_eigh_cusolver(shape, dtype):
    """n > 2 on the user path: handled by the cuSOLVER fallback in _linalg_eigh."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp)

    # Eigenvalues of a Hermitian matrix are real even for complex inputs.
    utils.gems_assert_close(res_out[0], ref_out[0], res_out[0].dtype)
    # cuSOLVER path: relax tolerance to absorb the CPU round-trip + TF32 recon.
    _check_eigh_decomposition(inp, res_out[0], res_out[1], atol=1e-2)


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_COMPLEX_2X2_SHAPES,
    ids=[f"complex_2x2-{s[0]}x{s[1]}" for s in EIG_COMPLEX_2X2_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_linalg_eigh_complex_2x2(shape, dtype):
    """Complex n == 2: Triton can't specialise complex dtypes, so this routes
    to the cuSOLVER path (not the 2x2 analytical kernel). complex128 validation
    uses torch.testing.assert_close because FlagGems' gems_assert_close
    tolerance table has no complex128 entry (matches cholesky_solve/
    linalg_ldl_solve/linalg_cross)."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp)

    # Eigenvalues of a Hermitian matrix are real (float32/float64 here).
    utils.gems_assert_close(res_out[0], ref_out[0], res_out[0].dtype)
    _check_eigh_decomposition(inp, res_out[0], res_out[1], atol=1e-2)


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_BATCH_2X2_SHAPES,
    ids=[f"gpu_kernel_2x2-batch{s[0]}" for s in EIG_BATCH_2X2_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_batch_2x2_kernel(shape, dtype):
    """Batched n == 2 on the user path: each batch element hits the Triton kernel."""
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
    EIG_BATCH_CUSOLVER_SHAPES,
    ids=[f"cusolver-batch{s[-1]}" for s in EIG_BATCH_CUSOLVER_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_batch_cusolver(shape, dtype):
    """Batched n > 2 on the user path: cuSOLVER fallback."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp)

    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
    _check_eigh_decomposition(inp, res_out[0], res_out[1], atol=1e-2)


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_TRIVIAL_SHAPES,
    ids=[f"trivial_n{s[0]}" for s in EIG_TRIVIAL_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_trivial(shape, dtype):
    """n < 2 (0x0 / 1x1) on the user path: computed on device via host-side ops."""
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
    EIG_BATCH_TRIVIAL_SHAPES,
    ids=[f"trivial-batch_n{s[-1]}" for s in EIG_BATCH_TRIVIAL_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_batch_trivial(shape, dtype):
    """Batched n < 2 (0x0 / 1x1) on the user path: computed on device."""
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
    EIG_2X2_SHAPES,
    ids=[f"lowdtype_2x2-{s[0]}x{s[1]}" for s in EIG_2X2_SHAPES],
)
@pytest.mark.parametrize("dtype", EIG_2X2_LOWDTYPE)
def test_linalg_eigh_2x2_low_precision(shape, dtype):
    """n == 2 fp16/bf16: the 2x2 Triton path widens to fp32 on device and casts
    back. This is an enhancement over native torch.linalg.eigh (which raises
    NotImplementedError for Half/BFloat16). cuSOLVER reference is unavailable for
    these dtypes, so validate via reconstruction only."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp)

    # Eigenvalues/vec are in the input (low-precision) dtype after the cast back.
    _check_eigh_decomposition(inp, res_out[0], res_out[1], atol=1e-2)


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_TRIVIAL_COMPLEX_SHAPES,
    ids=[f"trivial_complex_n{s[0]}" for s in EIG_TRIVIAL_COMPLEX_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_linalg_eigh_trivial_complex(shape, dtype):
    """Complex n < 2 (0x0 / 1x1): routed to cuSOLVER (the on-device trivial
    path is real-floating-point only)."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp)

    _assert_close(res_out[0], ref_out[0], res_out[0].dtype)
    if res_out[1].numel() > 0:
        _check_eigh_decomposition(inp, res_out[0], res_out[1], atol=1e-2)


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_BATCH_CUSOLVER_COMPLEX_SHAPES,
    ids=[f"cusolver_complex-batch{s[-1]}" for s in EIG_BATCH_CUSOLVER_COMPLEX_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_linalg_eigh_batch_cusolver_complex(shape, dtype):
    """Batched complex n > 2 on the user path: cuSOLVER fallback."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp)

    _assert_close(res_out[0], ref_out[0], res_out[0].dtype)
    _check_eigh_decomposition(inp, res_out[0], res_out[1], atol=1e-2)


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_UPLO_U_SHAPES,
    ids=[f"uplo_u_n{s[0]}" for s in EIG_UPLO_U_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_uplo_upper(shape, dtype):
    """UPLO="U" on the cuSOLVER path: the lower/upper triangle selection is a
    core eigh parameter and must be honoured. Symmetric inputs make the two
    triangles equivalent, so this mainly guards that "U" is forwarded
    correctly (not silently dropped)."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp, UPLO="U")

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp, UPLO="U")

    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
    _check_eigh_decomposition(inp, res_out[0], res_out[1], atol=1e-2)


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_UPLO_U_COMPLEX_SHAPES,
    ids=[f"uplo_u_complex_n{s[0]}" for s in EIG_UPLO_U_COMPLEX_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_linalg_eigh_uplo_upper_complex(shape, dtype):
    """UPLO="U" on the complex cuSOLVER path."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp, UPLO="U")

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp, UPLO="U")

    _assert_close(res_out[0], ref_out[0], res_out[0].dtype)
    _check_eigh_decomposition(inp, res_out[0], res_out[1], atol=1e-2)


# ---------------------------------------------------------------------------
# Underlying entry: aten::_linalg_eigh (covers compute_v and the linalg_eigh
# public entry which mirrors the same paths).
# ---------------------------------------------------------------------------


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_2X2_SHAPES + EIG_CUSOLVER_SHAPES,
    ids=[f"ueigh-{s[0]}x{s[1]}" for s in EIG_2X2_SHAPES + EIG_CUSOLVER_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_underlying_linalg_eigh(shape, dtype):
    """Directly call aten::_linalg_eigh.default with compute_v=True.

    n == 2 delegates to the Triton kernel; n > 2 goes through cuSOLVER.
    """
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_w, ref_v = torch.ops.aten._linalg_eigh.default(ref_inp, "L", True)

    with flag_gems.use_gems():
        res_w, res_v = torch.ops.aten._linalg_eigh.default(inp, "L", True)

    utils.gems_assert_close(res_w, ref_w, dtype)
    _check_eigh_decomposition(inp, res_w, res_v, atol=1e-2)


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_CUSOLVER_SHAPES,
    ids=[f"ueigh_no_v_n{s[0]}" for s in EIG_CUSOLVER_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_underlying_linalg_eigh_no_vectors(shape, dtype):
    """compute_v=False returns eigenvalues only (empty eigenvectors)."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_w, _ = torch.ops.aten._linalg_eigh.default(ref_inp, "L", False)

    with flag_gems.use_gems():
        res_w, res_v = torch.ops.aten._linalg_eigh.default(inp, "L", False)

    utils.gems_assert_close(res_w, ref_w, dtype)
    # Eigenvectors tensor is empty when compute_v=False.
    assert res_v.numel() == 0


@pytest.mark.linalg_eigh
@pytest.mark.parametrize(
    "shape",
    EIG_2X2_SHAPES,
    ids=[f"ueigh_no_v_2x2-{s[0]}x{s[1]}" for s in EIG_2X2_SHAPES],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_underlying_linalg_eigh_no_vectors_2x2(shape, dtype):
    """compute_v=False with n == 2 real: the Triton delegation requires
    compute_v=True, so this goes through the cuSOLVER path in _linalg_eigh
    (distinct from the n==2 compute_v=True Triton delegation)."""
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_w, _ = torch.ops.aten._linalg_eigh.default(ref_inp, "L", False)

    with flag_gems.use_gems():
        res_w, res_v = torch.ops.aten._linalg_eigh.default(inp, "L", False)

    utils.gems_assert_close(res_w, ref_w, dtype)
    assert res_v.numel() == 0
