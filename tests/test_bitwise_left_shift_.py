import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Inplace bitwise ops require broadcast-compatible shapes where
# first dims are mapped according to broadcasting rules.
INPLACE_BITWISE_SHAPES = [
    ((512, 1024), (512, 1024)),
    ((256, 512), (1, 512)),
    ((256, 512), (256, 1)),
    ((1024,), ()),
]


@pytest.mark.bitwise_left_shift_
@pytest.mark.parametrize("shapes", INPLACE_BITWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES + [torch.uint8])
def test_bitwise_left_shift_(shapes, dtype):
    shape_a, shape_b = shapes
    inp1 = torch.randint(0, 100, shape_a, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    inp2 = torch.randint(0, 8, shape_b, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone())
    ref_inp2 = utils.to_reference(inp2)

    ref_inp1.bitwise_left_shift_(ref_inp2)
    with flag_gems.use_gems():
        inp1.bitwise_left_shift_(inp2)
    utils.gems_assert_close(inp1, ref_inp1, dtype)
    utils.gems_assert_close(inp2, ref_inp2, dtype)
