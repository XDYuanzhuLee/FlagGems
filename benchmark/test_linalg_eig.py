import pytest
import torch

from . import base


@pytest.mark.linalg_eig
def test_linalg_eig():
    bench = base.UnaryPointwiseBenchmark(
        op_name="linalg_eig",
        torch_op=torch.linalg_eig,
        dtypes=[torch.float32],
    )
    bench.run()
