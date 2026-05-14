import pytest
import torch

from . import base


@pytest.mark._unsafe_masked_index_put_accumulate
def test__unsafe_masked_index_put_accumulate():
    bench = base.UnaryPointwiseBenchmark(
        op_name="_unsafe_masked_index_put_accumulate",
        torch_op=torch._unsafe_masked_index_put_accumulate,
        dtypes=[torch.float32],
    )
    bench.run()
