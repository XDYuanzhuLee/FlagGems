import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

_UNSAFE_MASKED_INDEX_PUT_SHAPES = [
    (10,),
    (20,),
    (50,),
    (100,),
]
# Note: bfloat16 is excluded because tl.atomic_add doesn't support bfloat16
_UNSAFE_MASKED_INDEX_PUT_DTYPES = [torch.float16, torch.float32]


@pytest.mark.unsafe_masked_index_put_accumulate
@pytest.mark.parametrize("shape", _UNSAFE_MASKED_INDEX_PUT_SHAPES)
@pytest.mark.parametrize("dtype", _UNSAFE_MASKED_INDEX_PUT_DTYPES)
def test_accuracy__unsafe_masked_index_put_accumulate(shape, dtype):
    # Create input tensor
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    # Create mask with roughly 30% True values
    mask = torch.rand(shape) < 0.3
    mask = mask.to(flag_gems.device)

    # Create indices tensor with valid indices into input
    indices = torch.randint(
        0, max(shape[-1], 1), shape, dtype=torch.long, device=flag_gems.device
    )

    # Create values tensor
    values = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch._unsafe_masked_index_put_accumulate(
        ref_inp, mask, (indices,), values
    )
    with flag_gems.use_gems():
        res_out = torch._unsafe_masked_index_put_accumulate(
            inp, mask, (indices,), values
        )
