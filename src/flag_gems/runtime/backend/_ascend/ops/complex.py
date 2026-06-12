import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("complex"), key=["N2"])
@triton.jit
def complex_kernel_flat(
    real_ptr,  # float32/64
    imag_ptr,  # float32/64
    out_ptr,  # view 后的 float32/64
    N2,  # 2 * 元素个数
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N2

    src_idx = idx >> 1
    is_imag = (idx % 2) != 0

    r_val = tl.load(real_ptr + src_idx, mask=(src_idx < (N2 >> 1)), other=0.0)
    i_val = tl.load(imag_ptr + src_idx, mask=(src_idx < (N2 >> 1)), other=0.0)

    res = tl.where(is_imag, i_val, r_val)
    tl.store(out_ptr + idx, res, mask=mask)


def complex(real, imag):
    if real.dtype == torch.float32:
        out_dtype, base_dtype = torch.complex64, torch.float32
    else:
        out_dtype, base_dtype = torch.complex128, torch.float64

    # 1. 记录形状并展平 (解决 0 维 view 问题)
    orig_shape = real.shape
    real_flat = real.reshape(-1).contiguous()
    imag_flat = imag.reshape(-1).contiguous()

    N = real_flat.numel()
    N2 = 2 * N

    # 2. 分配空间并创建浮点视图 (解决 complex 指针签名问题)
    out_flat = torch.empty(N, dtype=out_dtype, device=real.device)
    out_view = out_flat.view(base_dtype)

    # 3. 设置 Grid
    def grid(meta):
        return (triton.cdiv(N2, meta["BLOCK_SIZE"]),)

    # 4. 启动 Kernel
    with torch_device_fn.device(real.device):
        complex_kernel_flat[grid](
            real_flat,
            imag_flat,
            out_view,
            N2,
        )

    # 5. 还原形状
    return out_flat.reshape(orig_shape)
