import pytest
import torch

from . import base


@pytest.mark._pin_memory
def test__pin_memory():
    bench = base.UnaryPointwiseBenchmark(
        op_name="_pin_memory",
        torch_op=torch._pin_memory,
        dtypes=[torch.float32],
    )
    bench.run()
