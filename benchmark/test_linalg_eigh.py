# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from . import base

# Benchmark shapes for linalg_eigh.
# Only n == 2 is benchmarked; the op raises NotImplementedError for n > 2.

# n == 2: Triton `_eig_2x2_kernel` (on device, CUDA-graph compatible).
EIG_BENCHMARK_2X2_SHAPES = [(2, 2)]


class LinalgEighBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = EIG_BENCHMARK_2X2_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            A = torch.randn(shape, dtype=cur_dtype, device=self.device)
            A = (A + A.transpose(-2, -1)) / 2
            yield A,


@pytest.mark.linalg_eigh
def test_linalg_eigh():
    bench = LinalgEighBenchmark(
        op_name="linalg_eigh",
        torch_op=torch.linalg.eigh,
        # linalg_eigh only supports float32/float64 on GPU
        dtypes=[torch.float32],
    )
    bench.run()
