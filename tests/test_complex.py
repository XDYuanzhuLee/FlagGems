import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def is_float64_supported():
    return flag_gems.device != "npu"


@pytest.mark.complex
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        pytest.param(
            torch.float64,
            marks=pytest.mark.skipif(
                not is_float64_supported(),
                reason="NPU Triton backend does not support float64 yet",
            ),
        ),
    ],
)
def test_complex(shape, dtype):
    real = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    imag = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_real = utils.to_reference(real)
    ref_imag = utils.to_reference(imag)
    ref_out = torch.complex(ref_real, ref_imag)

    with flag_gems.use_gems():
        res_out = torch.complex(real, imag)

    utils.gems_assert_equal(res_out.cpu(), ref_out.cpu())


@pytest.mark.complex
@pytest.mark.parametrize("dtype", [torch.float32])
def test_complex_non_contiguous(dtype):
    shape = (1024, 2)
    real = torch.randn(shape, dtype=dtype, device=flag_gems.device)[:, 0]
    imag = torch.randn(shape, dtype=dtype, device=flag_gems.device)[:, 1]

    ref_real = utils.to_reference(real)
    ref_imag = utils.to_reference(imag)
    ref_out = torch.complex(ref_real, ref_imag)

    with flag_gems.use_gems():
        res_out = torch.complex(real, imag)

    utils.gems_assert_equal(res_out.cpu(), ref_out.cpu())
