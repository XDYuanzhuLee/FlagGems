import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("addbmm"),
    key=["M", "N", "K"],
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addbmm_kernel(
    input_tensor,  # self tensor (M, N)
    batch1,  # batch1 tensor (batch, M, K)
    batch2,  # batch2 tensor (batch, K, N)
    output,  # output tensor (M, N)
    alpha,
    beta,
    M,
    N,
    K,
    batch_size,
    stride_input,
    stride_batch1_b,
    stride_batch1_m,
    stride_batch1_k,
    stride_batch2_b,
    stride_batch2_k,
    stride_batch2_n,
    stride_output_m,
    stride_output_n,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
):
    pidx = tl.program_id(0)
    pidy = tl.program_id(1)

    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        gridx = tl.num_programs(0)
        gridy = tl.num_programs(1)
        pid = pidx + pidy * gridx
        num_CTA_per_group = gridy * GROUP_M
        group_id = pid // num_CTA_per_group
        inner_group_id = pid % num_CTA_per_group
        GROUP_SIZE = tl.where(
            (group_id * GROUP_M + GROUP_M) > gridx, gridx % GROUP_M, GROUP_M
        )
        pid_m = group_id * GROUP_M + inner_group_id % GROUP_SIZE
        pid_n = inner_group_id // GROUP_SIZE

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    if not DIVISIBLE_M:
        mask_m = offs_m < M
    if not DIVISIBLE_N:
        mask_n = offs_n < N

    # Compute accumulator for all batches
    accumulator = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    for batch_idx in range(batch_size):
        # Offsets for current batch
        batch1_ptrs = (
            batch1
            + batch_idx * stride_batch1_b
            + offs_m[:, None] * stride_batch1_m
            + offs_k[None, :] * stride_batch1_k
        )
        batch2_ptrs = (
            batch2
            + batch_idx * stride_batch2_b
            + offs_k[:, None] * stride_batch2_k
            + offs_n[None, :] * stride_batch2_n
        )

        num_iters = tl.cdiv(K, TILE_K)
        tile_accumulator = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

        for _ in range(num_iters):
            if DIVISIBLE_K:
                if DIVISIBLE_M:
                    mask_a = None
                else:
                    mask_a = mask_m[:, None]
                if DIVISIBLE_N:
                    mask_b = None
                else:
                    mask_b = mask_n[None, :]
            else:
                mask_k = offs_k < K
                if DIVISIBLE_M:
                    mask_a = mask_k[None, :]
                else:
                    mask_a = mask_m[:, None] & mask_k[None, :]
                if DIVISIBLE_N:
                    mask_b = mask_k[:, None]
                else:
                    mask_b = mask_k[:, None] & mask_n[None, :]

            a = tl.load(batch1_ptrs, mask=mask_a)
            b = tl.load(batch2_ptrs, mask=mask_b)

            tile_accumulator += tl.dot(a, b, allow_tf32=False)

            offs_k += TILE_K
            batch1_ptrs += TILE_K * stride_batch1_k
            batch2_ptrs += TILE_K * stride_batch2_k

        accumulator += tile_accumulator

    # Final output: beta * input + alpha * accumulator
    input_ptrs = input_tensor + offs_m[:, None] * stride_input + offs_n[None, :]
    output_ptrs = (
        output + offs_m[:, None] * stride_output_m + offs_n[None, :] * stride_output_n
    )

    if DIVISIBLE_M and DIVISIBLE_N:
        mask_c = None
    elif DIVISIBLE_M and not DIVISIBLE_N:
        mask_c = mask_n[None, :]
    elif not DIVISIBLE_M and DIVISIBLE_N:
        mask_c = mask_m[:, None]
    else:
        mask_c = mask_m[:, None] & mask_n[None, :]

    input_val = tl.load(input_ptrs, mask=mask_c)
    out = accumulator * alpha + input_val * beta
    out = out.to(input_val.dtype)
    tl.store(output_ptrs, out, mask=mask_c)


def addbmm_(self, batch1, batch2, beta=1.0, alpha=1.0):
    logger.debug("ILUVATAR GEMS addbmm_")

    batch, M, K = batch1.shape
    _, _, N = batch2.shape

    assert batch1.shape[0] == batch2.shape[0], "Batch dim mismatch"
    assert batch1.shape[2] == batch2.shape[1], "K dim mismatch"
    assert self.shape == (M, N), f"self shape mismatch: expected ({M}, {N}), got {self.shape}"

    batch1 = batch1.contiguous()
    batch2 = batch2.contiguous()

    # In-place update
    with torch_device_fn.device(self.device):
        addbmm_kernel[lambda meta: (triton.cdiv(meta["M"], meta["TILE_M"]),
                                   triton.cdiv(meta["N"], meta["TILE_N"]))](
            self,
            batch1,
            batch2,
            self,  # output is self (in-place)
            alpha,
            beta,
            M,
            N,
            K,
            batch,
            self.stride(0),
            batch1.stride(0),
            batch1.stride(1),
            batch1.stride(2),
            batch2.stride(0),
            batch2.stride(1),
            batch2.stride(2),
            self.stride(0),
            self.stride(1),
        )

    return self