import pytest
import torch

from . import base


@pytest.mark.linalg_inv
def test_linalg_inv():
    bench = base.UnaryPointwiseBenchmark(
        op_name="linalg_inv",
        torch_op=torch.linalg_inv,
        dtypes=[torch.float32],
    )
    bench.run()
