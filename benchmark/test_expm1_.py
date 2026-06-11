import pytest
import torch

from . import base, consts


@pytest.mark.expm1_
def test_expm1_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="expm1_",
        torch_op=torch.expm1_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
