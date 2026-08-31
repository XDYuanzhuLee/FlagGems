import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# He_n(x) is computed in float32 intermediates regardless of input dtype, so
# both float32 and float64 results carry float32-precision error. gems_assert_close
# applies atol plus rtol=RESOLUTION[dtype]; rtol covers the large-value tail
# (|He_n(x)| up to ~1e6) and atol bounds the residual |res-ref| - rtol*|ref|,
# which reaches ~0.17 (float32) / ~0.15 (float64) at scalar n=10. The Iluvatar
# kernel uses a fully explicit polynomial expansion that accrues slightly more
# truncation error than the recurrence, so it keeps a wider float32 atol.
if flag_gems.vendor_name == "iluvatar":
    ATOL = {torch.float32: 2.0, torch.float64: 0.5}
else:
    ATOL = {torch.float32: 1.0, torch.float64: 0.5}


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
