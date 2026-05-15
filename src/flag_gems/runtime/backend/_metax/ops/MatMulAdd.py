import logging

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
    configs=runtime.get_tuned_config("matmuladd"),
    key=["M", "N", "K"],
)
@triton.jit
def matmuladd_kernel(
    a_ptr,
    b_ptr,
    i_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
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
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    i_ptrs = i_ptr + stride_im * offs_cm[:, None] + stride_in * offs_cn[None, :]
    bias = tl.load(i_ptrs, mask=c_mask, other=0.0)

    accumulator = accumulator + bias
    c = accumulator.to(bias.dtype)
    tl.store(c_ptrs, c, mask=c_mask)


def matmuladd(mat1, mat2, bias):
    """Computes matmul(mat1, mat2) + bias.

    This is a fused operation equivalent to:
        torch.matmul(mat1, mat2) + bias

    Args:
        mat1: First input tensor of shape (M, K) or (*, M, K)
        mat2: Second input tensor of shape (K, N) or (*, K, N)
        bias: Bias tensor of shape (M, N) or (M, N) or broadcastable

    Returns:
        Output tensor of shape (M, N) or (*, M, N)
    """
    logger.debug("METAX GEMS MATMULADD")

    assert mat1.shape[-1] == mat2.shape[-2], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[-2], mat2.shape[-1])
    ), "Incompatible input shape"

    # Handle batch dimensions
    if mat1.dim() > 2:
        # Batched matmul: (*, M, K) @ (*, K, N) -> (*, M, N)
        batch_dims = mat1.shape[:-2]
        M, K = mat1.shape[-2:]
        _, N = mat2.shape[-2:]

        # Flatten batch dimensions for computation
        mat1_flat = mat1.reshape(-1, M, K)
        mat2_flat = mat2.reshape(-1, K, N)
        bias_expanded = bias.expand(-1, M, N) if bias.dim() == 2 else bias.expand(*batch_dims, M, N)

        # Allocate output
        out_flat = torch.empty((mat1_flat.shape[0], M, N), device=mat1.device, dtype=mat1.dtype)

        # Process each batch
        for b in range(mat1_flat.shape[0]):
            mat1_b = mat1_flat[b].contiguous()
            mat2_b = mat2_flat[b].contiguous()
            bias_b = bias_expanded[b].contiguous()

            logger.debug(
                "METAX GEMS MATMULADD, [shape info]: [-, %s, %s, %s](batch, M, N, K), "
                "[A column-major]: %s, [B column-major]: %s, [bias column-major]: %s",
                M,
                N,
                K,
                mat1_b.stride(0) == 1,
                mat2_b.stride(0) == 1,
                bias_b.stride(0) == 1,
            )

            grid = lambda META: (
                triton.cdiv(M, META["BLOCK_SIZE_M"]),
                triton.cdiv(N, META["BLOCK_SIZE_N"]),
            )
            with torch_device_fn.device(mat1.device):
                matmuladd_kernel[grid](
                    mat1_b,
                    mat2_b,
                    bias_b,
                    out_flat[b],
                    M,
                    N,
                    K,
                    mat1_b.stride(0),
                    mat1_b.stride(1),
                    mat2_b.stride(0),
                    mat2_b.stride(1),
                    bias_b.stride(0),
                    bias_b.stride(1),
                    out_flat[b].stride(0),
                    out_flat[b].stride(1),
                )

        return out_flat.reshape(*batch_dims, M, N)
    else:
        # Non-batched case
        M, K = mat1.shape
        _, N = mat2.shape

        mat1 = mat1.contiguous()
        mat2 = mat2.contiguous()
        out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
        bias = bias.broadcast_to(out.shape).contiguous()

        logger.debug(
            "METAX GEMS MATMULADD, [shape info]: [-, %s, %s, %s](batch, M, N, K), "
            "[A column-major]: %s, [B column-major]: %s, [bias column-major]: %s",
            M,
            N,
            K,
            mat1.stride(0) == 1,
            mat2.stride(0) == 1,
            bias.stride(0) == 1,
        )

        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )
        with torch_device_fn.device(mat1.device):
            matmuladd_kernel[grid](
                mat1,
                mat2,
                bias,
                out,
                M,
                N,
                K,
                mat1.stride(0),
                mat1.stride(1),
                mat2.stride(0),
                mat2.stride(1),
                bias.stride(0),
                bias.stride(1),
                out.stride(0),
                out.stride(1),
            )
        return out