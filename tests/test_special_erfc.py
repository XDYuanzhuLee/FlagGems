import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_erfc
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_erfc(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    if dtype in (torch.float16, torch.bfloat16):
        ref_out = torch.ops.aten.special_erfc(ref_x.float()).to(dtype)
    else:
        ref_out = torch.ops.aten.special_erfc(ref_x)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.special_erfc(x)
    utils.gems_assert_close(act_out, ref_out, dtype, equal_nan=True)


@pytest.mark.erfc_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_erfc_(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    if dtype in (torch.float16, torch.bfloat16):
        ref_out = torch.ops.aten.erfc_(ref_x.float())
        ref_out = ref_out.to(dtype)
        ref_x = ref_x.to(dtype)
    else:
        ref_out = torch.ops.aten.erfc_(ref_x)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.erfc_(x)
    utils.gems_assert_close(act_out, ref_out, dtype, equal_nan=True)
    utils.gems_assert_close(x, ref_x, dtype, equal_nan=True)
