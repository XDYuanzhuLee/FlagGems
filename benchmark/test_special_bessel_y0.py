import pytest
import torch

from . import base


@pytest.mark.special_bessel_y0
def test_special_bessel_y0():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_bessel_y0",
        torch_op=torch.special.bessel_y0,
        # bessel_y0_cuda only supports float32/float64, not float16/bfloat16
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.special_bessel_y0_out
def test_special_bessel_y0_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="special_bessel_y0_out",
        torch_op=torch.special.bessel_y0,
        # bessel_y0_cuda only supports float32/float64, not float16/bfloat16
        dtypes=[torch.float32],
    )
    bench.run()
