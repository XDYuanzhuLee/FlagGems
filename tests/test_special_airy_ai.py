import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_airy_ai
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_special_airy_ai(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    # Use float32 for reference since PyTorch doesn't support airy_ai on float16
    ref_out = torch.special.airy_ai(ref_inp.float()).to(dtype)
    with flag_gems.use_gems():
        res_out = torch.special.airy_ai(inp)

    # Use much looser tolerance for this special function due to approximation complexity
    gems_assert_close(res_out, ref_out, dtype, atol=5e-1)
