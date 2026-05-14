import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.Cross_Attention
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_Cross_Attention(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.Cross_Attention(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.Cross_Attention(inp)
    utils.gems_assert_close(res_out, ref_out, dtype)
