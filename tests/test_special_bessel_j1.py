import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_bessel_j1
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_special_bessel_j1(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    ref_out = torch.special.bessel_j1(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.bessel_j1(inp)
    gems_assert_close(res_out, ref_out, dtype)
