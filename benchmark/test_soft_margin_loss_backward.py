import pytest
import torch

from . import base


@pytest.mark.soft_margin_loss_backward
def test_soft_margin_loss_backward():
    bench = base.UnaryPointwiseBenchmark(
        op_name="soft_margin_loss_backward",
        torch_op=torch.soft_margin_loss_backward,
        dtypes=[torch.float32],
    )
    bench.run()
