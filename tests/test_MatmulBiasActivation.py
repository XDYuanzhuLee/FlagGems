import pytest
import torch

import flag_gems

from .accuracy_utils import FLOAT_DTYPES as ORIG_FLOAT_DTYPES
from .accuracy_utils import gems_assert_close, to_reference
from .conftest import QUICK_MODE

if QUICK_MODE:
    # Minimal shapes for fast CI validation
    MNK_SHAPES = [
        (1, 1, 32),
    ]
    # Use minimal dtype for fast CI; full dtype matrix is tested in non-QUICK mode
    FLOAT_DTYPES = [torch.float32]
else:
    # Representative matmul shapes covering small, medium, and unbalanced dimensions
    MNK_SHAPES = [
        (1, 1, 32),
        (15, 160, 1024),
        (495, 5333, 71),
    ]
    FLOAT_DTYPES = ORIG_FLOAT_DTYPES


@pytest.mark.MatmulBiasActivation
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_MatmulBiasActivation(M, N, K, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skipping fp32 MatmulBiasActivation test on tsingmicro platform")

    input_tensor = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)

    ref_input = to_reference(input_tensor, True)
    ref_weight = to_reference(weight, True)
    ref_bias = to_reference(bias, True)

    ref_out = torch.relu(torch.mm(ref_input, ref_weight) + ref_bias)
    with flag_gems.use_gems():
        from flag_gems.ops.MatmulBiasActivation import MatmulBiasActivation

        res_out = MatmulBiasActivation(input_tensor, weight, bias)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)
