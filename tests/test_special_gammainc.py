import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_gammainc
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# float32 only: gammainc series expansion is numerically unstable in lower precisions
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_gammainc(shape, dtype):
    # Use positive values for gammainc as it's defined for non-negative inputs
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 10 + 0.1
    y = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 10 + 0.1

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.special.gammainc(ref_x, ref_y)

    with flag_gems.use_gems():
        res_out = torch.special.gammainc(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_gammainc_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# float32 only: gammainc series expansion is numerically unstable in lower precisions
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_gammainc_out(shape, dtype):
    # Use positive values for gammainc as it's defined for non-negative inputs
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 10 + 0.1
    y = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 10 + 0.1

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out_buf = torch.empty(shape, dtype=ref_x.dtype, device=ref_x.device)
    ref_out = torch.ops.aten.special_gammainc.out(ref_x, ref_y, out=ref_out_buf)

    res_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.special_gammainc.out(x, y, out=res_out_buf)

    utils.gems_assert_close(res_out, ref_out, dtype)
