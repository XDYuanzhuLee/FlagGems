import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def get_test_dtypes():
    try:
        configs = flag_gems.runtime.get_tuned_config("complex")
        if configs and "dtypes" in configs[0]:
            mapping = {"float32": torch.float32, "float64": torch.float64}
            return [mapping[d] for d in configs[0]["dtypes"] if d in mapping]
    except Exception:
        pass
    return [torch.float32]


@pytest.mark.complex
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", get_test_dtypes())
def test_accuracy_complex(shape, dtype):
    real = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    imag = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_real = utils.to_reference(real)
    ref_imag = utils.to_reference(imag)
    ref_out = torch.complex(ref_real, ref_imag)

    with flag_gems.use_gems():
        res_out = torch.complex(real, imag)

    utils.gems_assert_equal(res_out.cpu(), ref_out.cpu())


@pytest.mark.complex
@pytest.mark.parametrize("dtype", get_test_dtypes())
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
