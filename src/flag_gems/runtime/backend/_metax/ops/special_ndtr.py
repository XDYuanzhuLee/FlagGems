import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger("flag_gems." + __name__)
erf = tl_extra_shim.erf
exp = tl_extra_shim.exp


@libentry()
@triton.jit
def special_ndtr_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(input_ptr + offsets, mask=mask)

    # ndtr(x) = 0.5 * (1 + erf(x / sqrt(2)))
    sqrt2: tl.constexpr = 1.4142135623730951
    x_f32 = x.to(tl.float32)
    result = 0.5 * (1.0 + erf(x_f32 / sqrt2))

    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


class SpecialNdtr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        logger.debug("METAX GEMS SPECIAL_NDTR FORWARD")
        # Create output tensor with float32 dtype
        output = torch.empty(A.shape, dtype=torch.float32, device=A.device)

        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        with torch_device_fn.device(A.device):
            special_ndtr_kernel[grid](
                A, output, n_elements, BLOCK_SIZE=1024
            )

        # Convert to original dtype
        ctx.save_for_backward(output)
        return output.to(dtype=A.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("METAX GEMS SPECIAL_NDTR BACKWARD")
        (out,) = ctx.saved_tensors

        # Create output tensor with float32 dtype
        in_grad = torch.empty(out.shape, dtype=torch.float32, device=out.device)

        n_elements = in_grad.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        @triton.jit
        def special_ndtr_backward_kernel(
            output_ptr,
            grad_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            y = tl.load(output_ptr + offsets, mask=mask)
            dy = tl.load(grad_ptr + offsets, mask=mask)

            # ndtr'(x) = exp(-x^2/2) / sqrt(2*pi)
            inv_sqrt_2pi: tl.constexpr = 0.3989422804014327  # 1/sqrt(2*pi)
            x_f32 = y.to(tl.float32)
            dy_f32 = dy.to(tl.float32)
            exp_term = exp(-0.5 * x_f32 * x_f32)
            result = dy_f32 * exp_term * inv_sqrt_2pi

            tl.store(grad_ptr + offsets, result, mask=mask)

        with torch_device_fn.device(out.device):
            special_ndtr_backward_kernel[grid](
                out, in_grad, n_elements, BLOCK_SIZE=1024
            )

        return in_grad.to(dtype=out_grad.dtype)


def special_ndtr(A):
    return SpecialNdtr.apply(A)