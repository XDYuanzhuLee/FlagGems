import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("complex"), key=["N2"])
@triton.jit
def complex_kernel_flat(
    real_ptr,  # 输入 float32/64
    imag_ptr,  # 输入 float32/64
    out_ptr,  # 输出 float32/64 (View 后的)
    N2,  # 2 * 元素个数
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # 处理平铺后的索引 [0, 1, 2, 3...]
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N2

    # 计算原实部/虚部的索引 (idx // 2)
    src_idx = idx >> 1
    # 判定当前位置是该存实部还是虚部 (idx % 2 != 0)
    is_imag = (idx % 2) != 0

    # 加载数据
    # NVIDIA 对 Mask 处理很稳，直接加载
    r_val = tl.load(real_ptr + src_idx, mask=(src_idx < (N2 >> 1)), other=0.0)
    i_val = tl.load(imag_ptr + src_idx, mask=(src_idx < (N2 >> 1)), other=0.0)

    # 选择结果：奇数位存虚部，偶数位存实部
    res = tl.where(is_imag, i_val, r_val)

    # 线性写入，避开一切 KeyError 和编译器优化 Bug
    tl.store(out_ptr + idx, res, mask=mask)


def complex(real, imag):
    logger.debug("GEMS COMPLEX - NVIDIA ADAPTED")

    # 1. 确定类型映射
    if real.dtype == torch.float32:
        out_dtype, base_dtype = torch.complex64, torch.float32
    elif real.dtype == torch.float64:
        out_dtype, base_dtype = torch.complex128, torch.float64
    else:
        raise TypeError(f"complex() expected float32 or float64, but got {real.dtype}")

    # 2. 预处理：解决 0 维张量无法 View 的问题
    orig_shape = real.shape
    real_flat = real.reshape(-1).contiguous()
    imag_flat = imag.reshape(-1).contiguous()

    N = real_flat.numel()
    N2 = 2 * N

    # 3. 分配空间并创建浮点视图 (重点：避开 KeyError: 'complex64')
    out_flat = torch.empty(N, dtype=out_dtype, device=real.device)
    out_view = out_flat.view(base_dtype)

    # 4. 启动 Kernel
    def grid(meta):
        return (triton.cdiv(N2, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(real.device):
        complex_kernel_flat[grid](
            real_flat,
            imag_flat,
            out_view,
            N2,
        )

    # 5. 还原形状并返回
    return out_flat.reshape(orig_shape)
