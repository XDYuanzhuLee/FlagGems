import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle

rsqrt = tl_extra_shim.rsqrt

logger = logging.getLogger("flag_gems." + __name__)


# Maximum BLOCK_HW_SIZE to avoid exceeding Metax private memory limit (4KB/thread)
# With BLOCK_GROUP_SIZE up to 64, we have: 64 * 128 float32 = 32KB (exceeds)
# With BLOCK_GROUP_SIZE up to 8 (typical for groupnorm), we have: 8 * 128 = 4KB (at limit)
MAX_BLOCK_HW = 128


@libentry()
@triton.jit(do_not_specialize=["eps"])
def group_norm_kernel(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    group_size,
    C,
    HW,
    num_groups,
    eps,
    BLOCK_GROUP_SIZE: tl.constexpr,
    BLOCK_HW_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    group = pid % num_groups
    num_elements = group_size * HW
    group_offset = tl.arange(0, BLOCK_GROUP_SIZE)
    hw_offset = tl.arange(0, BLOCK_HW_SIZE)

    wb_offset = group * group_size + group_offset
    wb_mask = wb_offset < C

    # Accumulator for mean and variance
    sum_val = tl.zeros([BLOCK_GROUP_SIZE, BLOCK_HW_SIZE], dtype=tl.float32)

    # Process HW in tiles
    for off in range(0, HW, BLOCK_HW_SIZE):
        hw_offsets = off + hw_offset
        hw_mask = hw_offsets < HW

        xy_offset = pid * num_elements + group_offset[:, None] * HW + hw_offsets[None, :]
        xy_mask = wb_offset[:, None] < C and hw_mask[None, :]

        X_ptr = X + xy_offset
        X_val = tl.load(X_ptr, mask=xy_mask, other=0.0).to(tl.float32)
        sum_val = tl.where(xy_mask, sum_val + X_val, sum_val)

    # Compute mean
    mean = tl.sum(sum_val) / num_elements

    # Compute variance
    var_sum = tl.zeros([BLOCK_GROUP_SIZE, BLOCK_HW_SIZE], dtype=tl.float32)
    for off in range(0, HW, BLOCK_HW_SIZE):
        hw_offsets = off + hw_offset
        hw_mask = hw_offsets < HW

        xy_offset = pid * num_elements + group_offset[:, None] * HW + hw_offsets[None, :]
        xy_mask = wb_offset[:, None] < C and hw_mask[None, :]

        X_ptr = X + xy_offset
        X_val = tl.load(X_ptr, mask=xy_mask, other=0.0).to(tl.float32)
        x = tl.where(xy_mask, X_val - mean, 0.0)
        var_sum = tl.where(xy_mask, var_sum + x * x, var_sum)

    var = tl.sum(var_sum) / num_elements
    rstd = rsqrt(var + eps)

    # Compute output
    y_sum = tl.zeros([BLOCK_GROUP_SIZE, BLOCK_HW_SIZE], dtype=tl.float32)
    for off in range(0, HW, BLOCK_HW_SIZE):
        hw_offsets = off + hw_offset
        hw_mask = hw_offsets < HW

        xy_offset = pid * num_elements + group_offset[:, None] * HW + hw_offsets[None, :]
        xy_mask = wb_offset[:, None] < C and hw_mask[None, :]

        X_ptr = X + xy_offset
        X_val = tl.load(X_ptr, mask=xy_mask, other=0.0).to(tl.float32)
        x = tl.where(xy_mask, X_val - mean, 0.0)
        x_hat = x * rstd

        if W is None:
            weight = 1.0
        else:
            weight = tl.load(W + wb_offset, mask=wb_mask, other=0.0).to(tl.float32)[:, None]
        if B is None:
            bias = 0.0
        else:
            bias = tl.load(B + wb_offset, mask=wb_mask, other=0.0).to(tl.float32)[:, None]
        Y_val = x_hat * weight + bias

        Y_ptr = Y + xy_offset
        tl.store(Y_ptr, Y_val, mask=xy_mask)

    Mean_ptr = Mean + pid
    Rstd_ptr = Rstd + pid
    tl.store(Mean_ptr, mean)
    tl.store(Rstd_ptr, rstd)


@libentry()
@triton.jit
def group_norm_backward_kernel(
    grad_y,
    X,
    W,
    Mean,
    Rstd,
    num_groups,
    group_size,
    grad_x,
    C,
    HW,
    BLOCK_GROUP_SIZE: tl.constexpr,
    BLOCK_HW_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    group = pid % num_groups
    num_elements = group_size * BLOCK_HW_SIZE

    group_offset = tl.arange(0, BLOCK_GROUP_SIZE)
    hw_offset = tl.arange(0, BLOCK_HW_SIZE)
    wb_offset = group * group_size + group_offset

    wb_mask = wb_offset < C

    xy_offset = pid * num_elements + group_offset[:, None] * HW + hw_offset[None, :]
    xy_mask = wb_offset[:, None] < C and hw_offset[None, :] < HW

    Mean_ptr = Mean + pid
    Rstd_ptr = Rstd + pid
    X_ptr = X + xy_offset
    dY_ptr = grad_y + xy_offset
    dX_ptr = grad_x + xy_offset

    rstd = tl.load(Rstd_ptr).to(tl.float32)
    mean = tl.load(Mean_ptr).to(tl.float32)
    dY_val = tl.load(dY_ptr, mask=xy_mask, other=0.0).to(tl.float32)
    X_val = tl.load(X_ptr, mask=xy_mask, other=0.0).to(tl.float32)

    if W is None:
        weight = 1
    else:
        weight = tl.load(W + wb_offset, mask=wb_mask, other=0.0).to(tl.float32)[:, None]

    dx_hat = weight * dY_val

    x = tl.where(xy_mask, X_val - mean, 0.0)

    grad_std = tl.sum(dx_hat * x)
    grad_var = grad_std * -(0.5 * rstd * rstd * rstd) / (HW * group_size)
    grad_distance = 2 * x * grad_var
    grad_centered_mean = dx_hat * rstd + grad_distance
    grad_mean = -tl.sum(grad_centered_mean) / num_elements
    grad_X = grad_centered_mean + grad_mean
    tl.store(dX_ptr, grad_X, mask=xy_mask)


@libentry()
@triton.jit
def weight_bias_backward_kernel(
    dY,
    X,
    Mean,
    Rstd,
    dW,
    dB,
    num_groups,
    group_size,
    N,
    C,
    HW,
    BLOCK_N: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tle.program_id(0)
    group = pid // group_size
    n_offset = tl.arange(0, BLOCK_N)
    hw_offset = tl.arange(0, BLOCK_HW)
    xy_mask = n_offset[:, None] < N and hw_offset[None, :] < HW
    mr_mask = n_offset < N

    mean_ptr = Mean + group + n_offset * num_groups
    rstd_ptr = Rstd + group + n_offset * num_groups

    dY_ptr = dY + pid * BLOCK_HW + n_offset[:, None] * C * HW + hw_offset[None, :]
    x_ptr = X + pid * BLOCK_HW + n_offset[:, None] * C * HW + hw_offset[None, :]

    grad_y = tl.load(dY_ptr, mask=xy_mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr, mask=xy_mask, other=0.0)
    x_f32 = x.to(tl.float32)
    mean = tl.load(mean_ptr, mask=mr_mask, other=0.0).to(tl.float32)[:, None]
    rstd = tl.load(rstd_ptr, mask=mr_mask, other=0.0).to(tl.float32)[:, None]

    if dW is not None:
        dw = tl.sum((x_f32 - mean) * rstd * grad_y, 1)
        dw = tl.sum(dw)
        tl.store(dW + pid, dw.to(x.dtype))
    if dB is not None:
        db = tl.sum(grad_y, 1)
        db = tl.sum(db)
        tl.store(dB + pid, db.to(x.dtype))


class GroupNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, N, C, HW, num_groups, weight=None, bias=None, eps=1e-05):
        logger.debug("METAX GEMS GROUPNORM FORWARD")
        group_size = C // num_groups
        x = x.contiguous()
        if weight is not None:
            weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        y = torch.empty_like(x)
        mean = torch.empty((N, num_groups), dtype=x.dtype, device=x.device)
        rstd = torch.empty((N, num_groups), dtype=x.dtype, device=x.device)
        grid = (N * num_groups,)

        # Cap BLOCK_HW_SIZE to avoid exceeding Metax private memory limit (4KB/thread)
        block_hw_size = min(triton.next_power_of_2(HW), MAX_BLOCK_HW)

        with torch_device_fn.device(x.device):
            group_norm_kernel[grid](
                x,
                y,
                weight,
                bias,
                mean,
                rstd,
                group_size,
                C,
                HW,
                num_groups,
                eps,
                BLOCK_GROUP_SIZE=triton.next_power_of_2(C // num_groups),
                BLOCK_HW_SIZE=block_hw_size,
            )
        if x.requires_grad:
            ctx.save_for_backward(x, weight, bias, mean, rstd)
            ctx.num_groups = num_groups
            ctx.group_size = group_size
            ctx.N = N
            ctx.C = C
            ctx.HW = HW
        return y, mean, rstd

    @staticmethod
    def backward(ctx, y_grad, mean_grad, rstd_grad):
        logger.debug("METAX GEMS GROUPNORM BACKWARD")
        y_grad = y_grad.contiguous()
        (x, weight, bias, mean, rstd) = ctx.saved_tensors
        num_groups = ctx.num_groups
        group_size = ctx.group_size
        N = ctx.N
        C = ctx.C
        HW = ctx.HW
        x_grad = torch.empty_like(x)
        grid = (N * num_groups,)
        with torch_device_fn.device(x.device):
            group_norm_backward_kernel[grid](
                y_grad,
                x,
                weight,
                mean,
                rstd,
                num_groups,
                group_size,
                x_grad,
                C,
                HW,
                BLOCK_GROUP_SIZE=triton.next_power_of_2(C // num_groups),
                BLOCK_HW_SIZE=triton.next_power_of_2(HW),
            )
        if weight is None and bias is None:
            return x_grad, None, None, None, None, None, None, None

        weight_grad = None if weight is None else torch.empty_like(weight)
        bias_grad = None if bias is None else torch.empty_like(bias)
        with torch_device_fn.device(x.device):
            weight_bias_backward_kernel[(C, 1, 1)](
                y_grad,
                x,
                mean,
                rstd,
                weight_grad,
                bias_grad,
                num_groups,
                group_size,
                N,
                C,
                HW,
                BLOCK_N=triton.next_power_of_2(N),
                BLOCK_HW=triton.next_power_of_2(HW),
            )
        return x_grad, None, None, None, None, weight_grad, bias_grad, None


def group_norm(x, weight, bias, N, C, HW, num_groups, eps):
    return GroupNorm.apply(x, N, C, HW, num_groups, weight, bias, eps)
