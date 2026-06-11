import pytest
import torch

import flag_gems
from flag_gems.utils import shape_utils

from . import base


class ComplexBenchmark(base.GenericBenchmark):
    """
    参照 UnbindCopy 实现的 Complex Benchmark。
    使用 GenericBenchmark 可以完全自定义输入和指标，避开自动注入参数的 Bug。
    """

    def set_more_metrics(self):
        # 1. 显式指定只测量 gbps (带宽)，避开会导致崩溃的 tflops (计算量)
        return ["gbps"]

    def get_gbps(self, bench_fn_args, latency):
        # 2. 精确计算 IO 量：
        # 输入：Read Real (N bytes) + Read Imag (N bytes)
        # 输出：Write Complex (2N bytes)
        # 总 IO = 4 * size_in_bytes(real_tensor)
        real = bench_fn_args[0]
        io_amount = shape_utils.size_in_bytes(real) * 4
        return io_amount * 1e-9 / (latency * 1e-3)


def _complex_input_fn(shape, dtype, device):
    """
    精确控制输入生成，确保传给算子的只有 real 和 imag 两个 Tensor。
    """
    real = torch.randn(shape, dtype=dtype, device=device)
    imag = torch.randn(shape, dtype=dtype, device=device)
    yield real, imag


@pytest.mark.complex
def test_benchmark_complex():
    # 3. 使用完全自定义的 Benchmark 类
    bench = ComplexBenchmark(
        op_name="complex",
        torch_op=torch.complex,
        input_fn=_complex_input_fn,
        # NVIDIA 平台性能测试建议包含 float32 和 float64
        dtypes=[torch.float32, torch.float64],
    )
    # 绑定你的实现
    bench.set_gems(flag_gems.complex)
    bench.run()
