import pytest
import torch

from . import base, consts

# Benchmark shapes for replication_pad3d_backward
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


class ReplicationPad3dBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = REPLICATION_PAD3D_BACKWARD_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            for padding in REPLICATION_PAD3D_BACKWARD_PADDING:
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

                inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
                grad_output = torch.randn(
                    N, C, D_out, H_out, W_out, dtype=cur_dtype, device=self.device
                )
                yield grad_output, inp, padding


@pytest.mark.replication_pad3d_backward
def test_replication_pad3d_backward():
    bench = ReplicationPad3dBackwardBenchmark(
        op_name="replication_pad3d_backward",
        torch_op=torch.ops.aten.replication_pad3d_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
