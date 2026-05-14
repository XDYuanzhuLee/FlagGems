import pytest
import torch

from . import base


@pytest.mark.logcumsumexp
def test_logcumsumexp():
    bench = base.UnaryPointwiseBenchmark(
        op_name="logcumsumexp",
        torch_op=torch.logcumsumexp,
        dtypes=[torch.float32],
    )
    bench.run()
