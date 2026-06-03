import pytest
import torch

from . import base, consts


def _input_fn(b, m, n, k, dtype, device, b_column_major):
    input_tensor = torch.randn([m, k], dtype=dtype, device=device)
    weight = torch.randn([k, n], dtype=dtype, device=device)
    bias = torch.randn([n], dtype=dtype, device=device)
    yield input_tensor, weight, bias


@pytest.mark.MatmulBiasActivation
def test_MatmulBiasActivation():
    def torch_op(input, weight, bias):
        return torch.relu(torch.mm(input, weight) + bias)

    from flag_gems.ops.MatmulBiasActivation import MatmulBiasActivation

    bench = base.BlasBenchmark(
        input_fn=_input_fn,
        op_name="MatmulBiasActivation",
        torch_op=torch_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.gems_op = MatmulBiasActivation
    bench.run()
