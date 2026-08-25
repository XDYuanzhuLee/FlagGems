import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Hermite polynomials use unbounded inputs (randn); |He_n(x)| grows with n and
# |x|, reaching ~1e6 at n=10, so absolute error tracks the fp32 representability
# limit. gems_assert_close applies atol plus rtol=RESOLUTION[dtype]; rtol covers
# the large-value tail, so atol only bounds the residual max(|res-ref| -
# rtol*|ref|), which reaches ~0.17 at scalar n=10 for float32. atol is set with
# a ~6x margin to keep n=10 from flaking. float64 keeps input precision and is
# accurate enough that rtol alone covers every n. The Iluvatar kernel uses a
# fully explicit polynomial expansion that accrues more truncation error, so it
# keeps a wider float32 tolerance.
if flag_gems.vendor_name == "iluvatar":
    ATOL = {torch.float32: 2.0, torch.float64: 1e-2}
else:
    ATOL = {torch.float32: 1.0, torch.float64: 1e-3}


@pytest.mark.special_hermite_polynomial_he
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# CUDA does not support half/bfloat16 for this special function
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_hermite_polynomial_he(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # n is a tensor with small integer values (degree of polynomial)
    inp2 = torch.randint(0, 11, shape, dtype=torch.int64, device=flag_gems.device)

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2)
    # iluvatar 的原生 PyTorch CUDA kernel 不支持 int64 n → float 的 cast
    # (nvrtc ERROR_UNSUPPORTED_CAST)，参考计算强制走 CPU。
    if flag_gems.vendor_name == "iluvatar":
        ref_inp1 = ref_inp1.to("cpu")
        ref_inp2 = ref_inp2.to("cpu")

    ref_out = torch.special.hermite_polynomial_he(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.special.hermite_polynomial_he(inp1, inp2)

    if flag_gems.vendor_name == "iluvatar":
        res_out = res_out.to("cpu")
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=ATOL[dtype])

    # Also test scalar n path (n=0..10, where n=10 is the worst case)
    for n in range(0, 11):
        ref_out = torch.special.hermite_polynomial_he(ref_inp1, n)
        with flag_gems.use_gems():
            res_out = torch.special.hermite_polynomial_he(inp1, n)

        if flag_gems.vendor_name == "iluvatar":
            res_out = res_out.to("cpu")
        utils.gems_assert_close(
            res_out, ref_out, dtype, equal_nan=True, atol=ATOL[dtype]
        )
