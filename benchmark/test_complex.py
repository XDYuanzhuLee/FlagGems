import pytest
import torch

import flag_gems
from flag_gems.utils import shape_utils

from . import base


def get_supported_dtypes():
    try:
        configs = flag_gems.runtime.get_tuned_config("complex")
        if configs and "dtypes" in configs[0]:
            mapping = {"float32": torch.float32, "float64": torch.float64}
            return [mapping[d] for d in configs[0]["dtypes"] if d in mapping]
    except Exception:
        pass
    return (
        [torch.float32] if flag_gems.device == "npu" else [torch.float32, torch.float64]
    )


class ComplexBenchmark(base.GenericBenchmark):
    def set_more_metrics(self):
        return ["gbps"]

    def get_gbps(self, bench_fn_args, latency):
        real = bench_fn_args[0]
        io_amount = shape_utils.size_in_bytes(real) * 4
        return io_amount * 1e-9 / (latency * 1e-3)


def _complex_input_fn(shape, dtype, device):
    real = torch.randn(shape, dtype=dtype, device=device)
    imag = torch.randn(shape, dtype=dtype, device=device)
    yield real, imag


@pytest.mark.complex
def test_benchmark_complex():
    bench = ComplexBenchmark(
        op_name="complex",
        torch_op=torch.complex,
        input_fn=_complex_input_fn,
        dtypes=get_supported_dtypes(),
    )
    bench.set_gems(flag_gems.complex)
    bench.run()
