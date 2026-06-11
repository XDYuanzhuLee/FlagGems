import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# 5D shapes for replication_pad3d_backward: (N, C, D, H, W)
REPLICATION_PAD3D_BACKWARD_SHAPES = [
    (1, 2, 2, 2, 3),  # Simple case
    (2, 4, 8, 8, 16),  # Larger case
    (1, 1, 16, 16, 16),  # More cubic
]

REPLICATION_PAD3D_BACKWARD_PADDING = [
    (0, 0, 0, 0, 0, 0),  # No padding
    (1, 1, 0, 0, 0, 0),  # Padding only in width
    (1, 2, 0, 1, 0, 2),  # Padding in all dimensions
    (0, 0, 2, 2, 1, 1),  # Different padding
]


@pytest.mark.replication_pad3d_backward
@pytest.mark.parametrize("shape", REPLICATION_PAD3D_BACKWARD_SHAPES)
@pytest.mark.parametrize("padding", REPLICATION_PAD3D_BACKWARD_PADDING)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_replication_pad3d_backward(shape, padding, dtype):
    # Create input tensor
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    # Compute output shape after padding
    N, C, D, H, W = shape
    (
        pad_w_before,
        pad_w_after,
        pad_h_before,
        pad_h_after,
        pad_d_before,
        pad_d_after,
    ) = padding
    D_out = D + pad_d_before + pad_d_after
    H_out = H + pad_h_before + pad_h_after
    W_out = W + pad_w_before + pad_w_after

    # Create grad_output with the padded shape
    grad_output = torch.randn(
        N, C, D_out, H_out, W_out, dtype=dtype, device=flag_gems.device
    )

    # Reference implementation
    ref_inp = utils.to_reference(inp)
    ref_grad_output = utils.to_reference(grad_output)
    ref_out = torch.ops.aten.replication_pad3d_backward(
        ref_grad_output, ref_inp, padding
    )

    # FlagGems implementation
    with flag_gems.use_gems():
        res_out = torch.ops.aten.replication_pad3d_backward(grad_output, inp, padding)

    # The atomic_add operation causes precision loss when accumulating gradients
    if dtype in (torch.float16, torch.bfloat16):
        utils.gems_assert_close(res_out, ref_out, dtype, atol=0.05)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)
