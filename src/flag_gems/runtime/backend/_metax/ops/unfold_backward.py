import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)


@libentry()
@triton.jit
def _unfold_backward_kernel(
    grad_in_ptr,
    grad_out_ptr,
    numel_in,
    prod_after,
    L,
    size,
    step,
    D,
    inner_total,
    BLOCK: tl.constexpr,
):
    pid = tle.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel_in

    vals = tl.load(grad_in_ptr + offs, mask=mask, other=0)
    vals_f32 = vals.to(tl.float32)

    k = offs % size
    tmp1 = offs // size
    after_lin = tmp1 % prod_after
    tmp2 = offs // (prod_after * size)
    s = tmp2 % L
    before_lin = offs // inner_total

    pos = s * step + k

    out_id = ((before_lin * D) + pos) * prod_after + after_lin

    tl.atomic_add(grad_out_ptr + out_id, vals_f32, mask=mask)


def unfold_backward(
    grad_in: torch.Tensor, input_sizes, dim: int, size: int, step: int
) -> torch.Tensor:
    logger.debug("METAX GEMS UNFOLD BACKWARD")
    if step <= 0:
        raise ValueError("step must be > 0")

    if not isinstance(input_sizes, (list, tuple)):
        input_sizes = list(input_sizes)
    input_sizes = [int(s) for s in input_sizes]
    ndim = len(input_sizes)
    d = dim % ndim

    D = int(input_sizes[d])
    L = (D - int(size)) // int(step) + 1

    prod_after = 1
    for s_ in input_sizes[d + 1 :]:
        prod_after *= int(s_)
    inner_total = int(L) * int(prod_after) * int(size)

    device = grad_in.device
    grad_out_f32 = torch.zeros(input_sizes, dtype=torch.float32, device=device)

    numel_in = grad_in.numel()

    BLOCK = 128
    grid = lambda meta: (triton.cdiv(numel_in, meta["BLOCK"]),)

    with torch_device_fn.device(grad_in.device):
        _unfold_backward_kernel[grid](
            grad_in,
            grad_out_f32,
            numel_in,
            prod_after,
            L,
            size,
            step,
            D,
            inner_total,
            BLOCK=BLOCK,
        )

    if grad_in.dtype != torch.float32:
        return grad_out_f32.to(grad_in.dtype)
    return grad_out_f32