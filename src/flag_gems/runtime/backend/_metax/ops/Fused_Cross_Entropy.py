import logging

import torch
import triton
import triton.language as tl
from torch.nn import _reduction as _Reduction

from flag_gems import runtime
from flag_gems.fused.cross_entropy_loss import (
    celoss_indices_bwd,
    celoss_indices_kernel,
    celoss_indices_smooth_bwd,
    celoss_indices_smooth_kernel,
    celoss_probability_bwd,
    celoss_probability_kernel,
    sum_and_scale,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger("flag_gems." + __name__)


class FusedCrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, target, weight, reduction, ignore_index, label_smoothing):
        logger.debug("METAX GEMS FUSED_CROSS_ENTROPY LOSS")

        shape = list(inp.shape)
        dim = inp.ndim
        N = 1 if dim == 1 else shape[0]
        C = shape[0] if dim == 1 else shape[1]
        D = inp.numel() // N // C
        axis = 0 if dim == 1 else 1
        del shape[axis]

        inp = inp.contiguous()
        tgt = target.contiguous()
        weight = weight.contiguous() if weight is not None else None
        out = torch.empty(shape, dtype=torch.float32, device=inp.device)
        grid = lambda meta: (triton.cdiv(D, meta["BLOCK_D"]), N)

        if tgt.ndim == dim:
            # target probabilities
            with torch_device_fn.device(inp.device):
                celoss_probability_kernel[grid](
                    inp,
                    tgt,
                    weight,
                    out,
                    label_smoothing,
                    C,
                    D,
                )
        elif label_smoothing == 0:
            # target indices
            w_tgt = torch.empty(shape, dtype=torch.float32, device=inp.device)
            with torch_device_fn.device(inp.device):
                celoss_indices_kernel[grid](
                    inp,
                    tgt,
                    weight,
                    out,
                    w_tgt,
                    ignore_index,
                    C,
                    D,
                )
        else:
            w_tgt = torch.empty(shape, dtype=torch.float32, device=inp.device)
            with torch_device_fn.device(inp.device):
                celoss_indices_smooth_kernel[grid](
                    inp,
                    tgt,
                    weight,
                    out,
                    w_tgt,
                    ignore_index,
                    label_smoothing,
                    C,
                    D,
                )

        if reduction == 1:  # MEAN
            out_reduce = torch.empty([], dtype=inp.dtype, device=inp.device)
            if tgt.ndim == dim:
                sum_and_scale[(1,)](out, out_reduce, N * D, False, scale=N * D)
            else:
                wgt_sum = torch.empty([], dtype=torch.float32, device=inp.device)
                sum_and_scale[(1,)](
                    out, out_reduce, N * D, True, scale=w_tgt, mean_num=wgt_sum
                )
            out = out_reduce
        elif reduction == 2:  # SUM
            out_reduce = torch.empty([], dtype=inp.dtype, device=inp.device)
            sum_and_scale[(1,)](out, out_reduce, N * D, False)
            out = out_reduce

        if inp.requires_grad:
            ctx.save_for_backward(inp, tgt, weight)
            ctx.N = N
            ctx.C = C
            ctx.D = D
            ctx.ignore_index = ignore_index
            ctx.label_smoothing = label_smoothing
            ctx.shape = shape
            ctx.mean_num = 1
            if reduction == 1:
                ctx.mean_num = N * D if tgt.ndim == dim else wgt_sum

        return out.to(inp.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("METAX GEMS FUSED_CROSS_ENTROPY LOSS VJP")

        inp, tgt, weight = ctx.saved_tensors
        N = ctx.N
        C = ctx.C
        D = ctx.D
        ignore_index = ctx.ignore_index
        label_smoothing = ctx.label_smoothing
        mean_num = (
            1 / ctx.mean_num.item()
            if isinstance(ctx.mean_num, torch.Tensor)
            else 1 / ctx.mean_num
        )
        shape = ctx.shape

        out_grad = out_grad.broadcast_to(shape).contiguous()

        inp_grad = torch.zeros(inp.shape, dtype=inp.dtype, device=inp.device)
        grid = lambda meta: (triton.cdiv(D, meta["BLOCK_D"]), N)
        if tgt.ndim == inp.ndim:
            with torch_device_fn.device(inp_grad.device):
                celoss_probability_bwd[grid](
                    out_grad, inp, tgt, weight, inp_grad, label_smoothing, mean_num, C, D
                )
        elif label_smoothing == 0:
            with torch_device_fn.device(inp_grad.device):
                celoss_indices_bwd[grid](
                    out_grad, inp, tgt, weight, inp_grad, ignore_index, mean_num, C, D
                )
        else:
            with torch_device_fn.device(inp_grad.device):
                celoss_indices_smooth_bwd[grid](
                    out_grad,
                    inp,
                    tgt,
                    weight,
                    inp_grad,
                    ignore_index,
                    label_smoothing,
                    mean_num,
                    C,
                    D,
                )
        return inp_grad, None, None, None, None, None


def Fused_Cross_Entropy(
    inp, target, weight=None, reduction="mean", ignore_index=-100, label_smoothing=0.0
):
    """
    Fused Cross Entropy Loss function.

    This is a Metax-optimized implementation that wraps the core cross entropy
    loss kernels with device-specific handling.

    Args:
        inp: Input tensor of shape (N, C) or (N, C, D)
        target: Target tensor of shape (N,) or (N, D)
        weight: Optional weight tensor of shape (C,)
        reduction: Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'
        ignore_index: Specifies a target value that is ignored and does not
            contribute to the input gradient. Default: -100
        label_smoothing: Specifies the amount of label smoothing. Default: 0.0

    Returns:
        Tensor of the cross entropy loss.
    """
    return FusedCrossEntropyLoss.apply(
        inp,
        target,
        weight,
        _Reduction.get_enum(reduction),
        ignore_index,
        label_smoothing,
    )