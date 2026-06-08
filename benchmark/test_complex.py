import pytest
import torch

import flag_gems

from . import base


@pytest.mark.complex
def test_complex_benchmark():
    bench = base.BinaryPointwiseBenchmark(
        op_name="complex", torch_op=torch.complex, dtypes=[torch.float32]
    )
    bench.set_gems(flag_gems.complex)
    bench.run()
