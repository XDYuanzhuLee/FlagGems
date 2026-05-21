import json
import math
import os

import torch
import torch.nn.functional as F

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn

_COSINE_SIM_ENABLED = os.environ.get("GEMS_COSINE_SIM", "0") == "1"
_COSINE_SIM_FILE = os.environ.get("GEMS_COSINE_SIM_FILE", "")

if runtime.device.vendor_name == "kunlunxin":
    RESOLUTION = {
        torch.bool: 0,
        torch.uint8: 0,
        torch.int8: 0,
        torch.int16: 0,
        torch.int32: 0,
        torch.int64: 0,
        torch.float16: 1e-3,
        torch.float32: 1.3e-6,
        torch.bfloat16: 0.016,
        torch.float64: 1e-7,
        torch.complex32: 1e-3,
        torch.complex64: 1.3e-6,
    }
else:
    RESOLUTION = {
        torch.bool: 0,
        torch.uint8: 0,
        torch.int8: 0,
        torch.int16: 0,
        torch.int32: 0,
        torch.int64: 0,
        torch.float8_e4m3fn: 1e-3,
        torch.float8_e5m2: 1e-3,
        torch.float8_e4m3fnuz: 1e-3,
        torch.float8_e5m2fnuz: 1e-3,
        torch.float16: 1e-3,
        torch.float32: 1.3e-6,
        torch.bfloat16: 0.016,
        torch.float64: 1e-7,
        torch.complex32: 1e-3,
        torch.complex64: 1.3e-6,
    }


def _maybe_move_to_cpu(res, ref):
    if res.device.type == "cpu" or ref.device.type == "cpu":
        return res, ref

    required = res.numel() * res.element_size()

    free_mem = None
    try:
        free_mem, _ = torch_device_fn.mem_get_info(res.device)
    except RuntimeError:
        pass

    # torch.isclose allocates an auxiliary tensor roughly the size of the inputs,
    # so ensure we have enough headroom; otherwise compare on CPU.
    HUGE_TENSOR_BYTES = 1 << 30  # 1 GiB
    if (free_mem is not None and required >= free_mem) or (
        required >= HUGE_TENSOR_BYTES
    ):
        return res.cpu(), ref.cpu()
    return res, ref


def _to_real_flat_float(t):
    t = t.detach()
    if t.is_complex():
        t = torch.view_as_real(t)
    return t.flatten().float()


def _compute_cosine_similarity(res, ref):
    if res.numel() == 0:
        return 1.0
    res_flat = _to_real_flat_float(res)
    ref_flat = _to_real_flat_float(ref)
    finite_mask = torch.isfinite(res_flat) & torch.isfinite(ref_flat)
    if finite_mask.sum() == 0:
        return -1.0
    res_flat = res_flat[finite_mask]
    ref_flat = ref_flat[finite_mask]
    res_norm = torch.linalg.norm(res_flat)
    ref_norm = torch.linalg.norm(ref_flat)
    if res_norm == 0 and ref_norm == 0:
        return 1.0
    if res_norm == 0 or ref_norm == 0:
        return 0.0
    return F.cosine_similarity(res_flat.unsqueeze(0), ref_flat.unsqueeze(0)).item()


def _report_cosine_similarity(cos_sim, res):
    shape_str = str(list(res.shape))
    dtype_str = str(res.dtype)
    test_name = os.environ.get("PYTEST_CURRENT_TEST", "")
    line = (
        f"[COSINE_SIM] test={test_name} shape={shape_str}"
        f" dtype={dtype_str} cosine_similarity={cos_sim:.6f}"
    )
    print(line, flush=True)
    if _COSINE_SIM_FILE:
        entry = {
            "test": test_name,
            "shape": shape_str,
            "dtype": dtype_str,
            "cosine_similarity": cos_sim if math.isfinite(cos_sim) else str(cos_sim),
        }
        with open(_COSINE_SIM_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def assert_close(res, ref, dtype, equal_nan=False, reduce_dim=1, atol=1e-4):
    if dtype is None:
        dtype = torch.float32
    assert res.dtype == dtype
    ref = ref.to(dtype)
    res, ref = _maybe_move_to_cpu(res, ref)
    if _COSINE_SIM_ENABLED:
        _MAX_COSINE_ELEMENTS = 100_000_000  # ~400MB in float32
        if res.numel() <= _MAX_COSINE_ELEMENTS:
            cos_sim = _compute_cosine_similarity(res, ref)
            _report_cosine_similarity(cos_sim, res)
    rtol = RESOLUTION[dtype]
    torch.testing.assert_close(
        res, ref, atol=atol * reduce_dim, rtol=rtol, equal_nan=equal_nan
    )


def assert_equal(res, ref, equal_nan=False):
    torch.testing.assert_close(res, ref, atol=0, rtol=0, equal_nan=equal_nan)
