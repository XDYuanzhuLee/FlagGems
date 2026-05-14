import pytest
import torch

from . import base


@pytest.mark.binary_cross_entropy_with_logits
def test_binary_cross_entropy_with_logits():
    bench = base.UnaryPointwiseBenchmark(
        op_name="binary_cross_entropy_with_logits",
        torch_op=torch.binary_cross_entropy_with_logits,
        dtypes=[torch.float32],
    )
    bench.run()
