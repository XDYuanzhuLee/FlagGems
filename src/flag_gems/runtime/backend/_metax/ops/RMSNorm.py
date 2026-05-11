import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)


@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_kernel(
    out_ptr,
    INV_RMS,
    in_ptr,
    w_ptr,
    y_stride_r,
    y_stride_c,
    x_stride_r,
    x_stride_c,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(in_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        in_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = in_ptr.dtype.element_ty

    pid = tle.program_id(0)
    out_ptr += pid * y_stride_r
    in_ptr += pid * x_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(in_ptr + cols * x_stride_c, mask, other=0.0).to(cdtype)

    var = tl.sum(x * x, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms * w).to(cdtype)
    tl.store(out_ptr + cols * y_stride_c, y, mask=mask)
    tl.store(INV_RMS + pid, rrms)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("rms_norm_loop"),
    key=["N"],
)
@triton.jit(do_not_specialize=["eps"])
def rms_norm_loop_kernel(
    out_ptr,
    INV_RMS,
    in_ptr,
    w_ptr,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    if tl.constexpr(in_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        in_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = in_ptr.dtype.element_ty

    pid = tle.program_id(0)

    # Pass 1: compute sum(x^2) in chunks
    acc = tl.zeros((TILE_N,), dtype=tl.float32)
    num_steps = tl.cdiv(N, TILE_N)

    for step in range(0, num_steps - 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets).to(tl.float32)
        acc += x * x

    # last step with mask
    start_n = (num_steps - 1) * TILE_N
    n_offsets = start_n + tl.arange(0, TILE_N)
    mask = n_offsets < N
    x = tl.load(in_ptr + pid * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
    acc += x * x

    var = tl.sum(acc) / N
    rrms = 1 / tl.sqrt(var + eps)
    tl.store(INV_RMS + pid, rrms)

    # Pass 2: normalize in reverse order (better L2 cache reuse)
    prev_multiple = prev_multiple_of(N, TILE_N)

    # first reverse step with mask
    for start_n in range(0, TILE_N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(cdtype)
        w = tl.load(w_ptr + n_offsets, mask=mask, other=0.0)
        y = (x * rrms * w).to(cdtype)
        tl.store(out_ptr + pid * N + n_offsets, y, mask=mask)

    for start_n in range(TILE_N, N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            eviction_policy="evict_first",
        ).to(cdtype)
        w = tl.load(w_ptr + n_offsets)
        y = (x * rrms * w).to(cdtype)
        tl.store(out_ptr + pid * N + n_offsets, y)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_grad_dx_kernel(
    X,
    DY,
    INV_RMS,
    DX,
    W,
    dx_stride_r,
    dx_stride_c,
    x_stride_r,
    x_stride_c,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    DX += pid * dx_stride_r
    X += pid * x_stride_r
    DY += pid * x_stride_r
    INV_RMS += pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    inv_rms = tl.load(INV_RMS).to(tl.float32)
    dy = tl.load(DY + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)

    dy = dy * w

    normalized_buf = x * inv_rms
    row_sum_stats = tl.sum(normalized_buf * dy, axis=0)

    norm_val = normalized_buf / N
    dx = (dy - norm_val * row_sum_stats) * inv_rms

    tl.store(DX + cols * dx_stride_c, dx, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_grad_dx_loop_kernel(
    X,
    DY,
    INV_RMS,
    DX,
    W,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    pid = tle.program_id(0)
    X += pid * N
    DY += pid * N
    DX += pid * N
    INV_RMS += pid

    inv_rms_val = tl.load(INV_RMS).to(tl.float32)

    # First pass: compute sum(normalized_buf * dy)
    acc = tl.zeros((TILE_N,), dtype=tl.float32)
    num_steps = tl.cdiv(N, TILE_N)

    for step in range(0, num_steps - 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        x = tl.load(X + n_offsets).to(tl.float32)
        dy = tl.load(DY + n_offsets).to(tl.float32)
        w = tl.load(W + n_offsets).to(tl.float32)

        normalized = x * inv_rms_val
        dy = dy * w
        acc += normalized * dy

    # Last step with mask
    start_n = (num_steps - 1) * TILE_N
    n_offsets = start_n + tl.arange(0, TILE_N)
    mask = n_offsets < N
    x = tl.load(X + n_offsets, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + n_offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + n_offsets, mask=mask, other=0.0).to(tl.float32)

    normalized = x * inv_rms_val
    dy = dy * w
    acc += normalized * dy

    row_sum_stats = tl.sum(acc)

    # Second pass: compute dx
    for step in range(0, num_steps - 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        x = tl.load(X + n_offsets).to(tl.float32)
        dy = tl.load(DY + n_offsets).to(tl.float32)
        w = tl.load(W + n_offsets).to(tl.float32)

        normalized = x * inv_rms_val
        dy = dy * w
        norm_val = normalized / N
        dx = (dy - norm_val * row_sum_stats) * inv_rms_val

        tl.store(DX + n_offsets, dx)

    # Last step with mask
    start_n = (num_steps - 1) * TILE_N
    n_offsets = start_n + tl.arange(0, TILE_N)
    mask = n_offsets < N
    x = tl.load(X + n_offsets, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + n_offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + n_offsets, mask=mask, other=0.0).to(tl.float32)

    normalized = x * inv_rms_val
    dy = dy * w
    norm_val = normalized / N
    dx = (dy - norm_val * row_sum_stats) * inv_rms_val
    tl.store(DX + n_offsets, dx, mask=mask)


@libentry()
@triton.jit
def rms_norm_grad_dw_kernel(
    X,
    DY,
    INV_RMS,
    DW,
    dx_stride_r,
    dx_stride_c,
    x_stride_r,
    x_stride_c,
    M,
    N,
    ROW_BLOCK_SIZE: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
):
    row_pid = tle.program_id(0)
    col_pid = tle.program_id(1)

    row_start = row_pid * ROW_BLOCK_SIZE
    col_start = col_pid * COL_BLOCK_SIZE

    offset = row_start * x_stride_r + col_start * x_stride_c
    X += offset
    DY += offset
    INV_RMS += row_start

    rows = tl.arange(0, ROW_BLOCK_SIZE)
    cols = tl.arange(0, COL_BLOCK_SIZE)

    row_mask = (row_start + rows) < M
    col_mask = (col_start + cols) < N

    x = tl.load(
        X + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c,
        row_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    inv_rms = tl.load(INV_RMS + rows, row_mask, other=0.0).to(tl.float32)
    dy = tl.load(
        DY + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c,
        row_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    d_weight = x * dy * inv_rms[:, None]
    partial_dweight_sum = tl.sum(d_weight, axis=0)

    tl.store(
        DW + row_pid * N + col_start + cols,
        partial_dweight_sum,
        mask=col_mask,
    )


def rms_norm_backward(dy, x, inv_rms, normalized_shape, weight, eps=1e-5):
    logger.debug("METAX GEMS RMS_NORM BACKWARD")

    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)

    # Limit BLOCK_SIZE to avoid exceeding Metax private memory limits
    # For large N, use a loop-based approach
    if N <= 4096:
        BLOCK_SIZE = triton.next_power_of_2(N)
        x = x.contiguous()
        dy = dy.contiguous()
        weight = weight.contiguous()
        dx = torch.empty_like(x)

        with torch_device_fn.device(x.device):
            rms_norm_grad_dx_kernel[M,](
                x, dy, inv_rms, dx, weight,
                dx.stride(0), dx.stride(1),
                x.stride(0), x.stride(1), N, eps, BLOCK_SIZE
            )
    else:
        # For large N, use a tiled approach
        TILE_N = 1024
        x = x.contiguous()
        dy = dy.contiguous()
        weight = weight.contiguous()
        dx = torch.empty_like(x)

        with torch_device_fn.device(x.device):
            rms_norm_grad_dx_loop_kernel[M,](
                x, dy, inv_rms, dx, weight, N, eps, TILE_N
            )

    ROW_BLOCK_SIZE = 16
    COL_BLOCK_SIZE = 256
    row_block_num = triton.cdiv(M, ROW_BLOCK_SIZE)
    col_block_num = triton.cdiv(N, COL_BLOCK_SIZE)

    partial_buffer = torch.empty(
        (row_block_num, N), dtype=torch.float32, device=x.device
    )

    with torch_device_fn.device(x.device):
        rms_norm_grad_dw_kernel[row_block_num, col_block_num](
            x,
            dy,
            inv_rms,
            partial_buffer,
            N,
            1,
            N,
            1,
            M,
            N,
            ROW_BLOCK_SIZE,
            COL_BLOCK_SIZE,
        )
        dw = (
            torch.sum(partial_buffer, dim=0, dtype=torch.float32)
            .to(x.dtype)
            .reshape(-1)
        )

    return dx, dw


class RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps=1e-5):
        logger.debug("METAX GEMS RMS_NORM")
        # Compute the forward pass and inv_rms
        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)

        x_cont = x.contiguous()
        weight_cont = weight.contiguous()
        y = torch.empty_like(x)
        inv_rms = torch.empty((M,), device=x.device, dtype=torch.float32)

        if N <= 4096:
            BLOCK_SIZE = triton.next_power_of_2(N)
            rms_norm_kernel[M,](
                y, inv_rms, x_cont, weight_cont,
                y.stride(0), y.stride(1),
                x_cont.stride(0), x_cont.stride(1), N, eps, BLOCK_SIZE
            )
        else:
            rms_norm_loop_kernel[M,](y, inv_rms, x_cont, weight_cont, N, eps)

        ctx.save_for_backward(x, inv_rms, weight)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, inv_rms, weight = ctx.saved_tensors
        normalized_shape = ctx.normalized_shape
        eps = ctx.eps

        dx, dw = rms_norm_backward(dy, x, inv_rms, normalized_shape, weight, eps)
        return dx, None, dw, None


def rms_norm(x, normalized_shape, weight, eps=1e-5):
    return RmsNorm.apply(x, normalized_shape, weight, eps)