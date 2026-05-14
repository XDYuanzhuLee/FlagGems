import pytest
import torch

from . import base


@pytest.mark.Cross_Attention
def test_Cross_Attention():
    bench = base.UnaryPointwiseBenchmark(
        op_name="Cross_Attention",
        torch_op=torch.Cross_Attention,
        dtypes=[torch.float32],
    )
    bench.run()
