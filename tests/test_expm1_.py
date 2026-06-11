import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.expm1_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_expm1_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.expm1_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.expm1_(inp)

    assert res_out.data_ptr() == inp.data_ptr()
    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(inp, ref_inp, dtype, equal_nan=True)
