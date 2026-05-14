import pytest
import torch

from . import base


@pytest.mark.special_bessel_j1
def test_special_bessel_j1():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_bessel_j1",
        torch_op=torch.special_bessel_j1,
        dtypes=[torch.float32],
    )
    bench.run()
