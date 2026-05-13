import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("addbmm"),
    key=["M", "N", "K"],
)
@triton.heuristics(
    {
        "UPGRADE": lambda args: math.ceil(
            (args["M"] * args["N"]) / (args["BLOCK_SIZE_M"] * args["BLOCK_SIZE_N"])
        ).bit_length()
        > 32,
    }
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addbmm_kernel(
    a_ptr,
    b_ptr,
    i_ptr,
    c_ptr,
    alpha,
    beta,
    M,
    N,
    K,
    batch,
    stride_a_batch,
    stride_a_m,
    stride_b_batch,
    stride_b_k,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    UPGRADE: tl.constexpr,
):
    # Compute the output for one MxN tile by summing over batch dimension
    # a_ptr: pointer to batch1, shape (batch, M, K)
    # b_ptr: pointer to batch2, shape (batch, K, N)
    if UPGRADE:
        pid_m = tle.program_id(0)
        pid_n = tle.program_id(1)
    else:
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator for the sum over batch dimension
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over batch dimension and accumulate
    for batch_idx in range(0, batch):
        # a_ptrs: offset into batch1 for current batch element
        # batch1 shape: (batch, M, K), strides: (stride_a_batch, stride_a_m, 1)
        a_base = a_ptr + batch_idx * stride_a_batch
        a_ptrs = a_base + offs_am[:, None] * stride_a_m + offs_k[None, :] * 1

        # b_ptrs: offset into batch2 for current batch element
        # batch2 shape: (batch, K, N), strides: (stride_b_batch, stride_b_k, 1)
        b_base = b_ptr + batch_idx * stride_b_batch
        b_ptrs = b_base + offs_k[:, None] * stride_b_k + offs_bn[None, :] * 1

        # Load and compute dot product for this batch element
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(
                a_ptrs,
                mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N),
                other=0.0,
            )
            accumulator += tl.dot(a, b, allow_tf32=False)
            a_ptrs += BLOCK_SIZE_K * 1  # stride for K dimension is 1
            b_ptrs += BLOCK_SIZE_K * stride_b_k

    # Load input bias and compute final result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    i_ptrs = i_ptr + stride_im * offs_cm[:, None] + stride_in * offs_cn[None, :]
    bias = tl.load(i_ptrs, mask=c_mask, other=0.0)

    accumulator = accumulator * alpha + bias * beta
    c = accumulator.to(bias.dtype)
    tl.store(c_ptrs, c, mask=c_mask)


def addbmm(input, batch1, batch2, *, beta=1, alpha=1):
    logger.debug("METAX GEMS ADDBMM")

    batch, M, K = batch1.shape
    _, _, N = batch2.shape

    assert batch1.shape[0] == batch2.shape[0], "Batch dimension must match"
    assert batch1.shape[2] == batch2.shape[1], "K dimension must match for matmul"
    assert broadcastable_to(
        input.shape, (M, N)
    ), "Incompatible input shape"

    logger.debug(
        "METAX GEMS ADDBMM, [shape info]: [%s, %s, %s, %s](batch, M, N, K), "
        "[A column-major]: %s, [B column-major]: %s, [input column-major]: %s",
        batch,
        M,
        N,
        K,
        batch1.stride(1) == 1,
        batch2.stride(1) == 1,
        input.stride(0) == 1,
    )

    batch1 = batch1.contiguous()
    batch2 = batch2.contiguous()
    out = torch.empty((M, N), device=batch1.device, dtype=batch1.dtype)
    input = input.broadcast_to(out.shape).contiguous()

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    with torch_device_fn.device(batch1.device):
        addbmm_kernel[grid](
            batch1,
            batch2,
            input,
            out,
            alpha,
            beta,
            M,
            N,
            K,
            batch,
            batch1.stride(0),
            batch1.stride(1),
            batch2.stride(0),
            batch2.stride(1),
            input.stride(0),
            input.stride(1),
            out.stride(0),
            out.stride(1),
        )
    return out