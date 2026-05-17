import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems." + __name__)

lgamma = tl_extra_shim.lgamma


@libentry()
@triton.jit
def multigammaln_kernel_1d(
    input_ptr,
    output_ptr,
    p: tl.constexpr,
    N,
    stride_in,
    stride_on,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N
    input_ptrs = input_ptr + offs_n * stride_in
    a = tl.load(input_ptrs, mask=mask, other=0.0)

    # Compute log(pi) * p * (p-1) / 4
    log_pi = 1.1447298858494
    C = log_pi * p * (p - 1) / 4.0

    log_gamma_sum = tl.zeros_like(a)

    # Sum log(gamma(a - (i-1)/2)) for i = 1 to p
    # Using lgamma from tl_extra_shim
    for i in range(1, p + 1):
        x = a - (i - 1) / 2.0
        log_gamma = lgamma(x)
        log_gamma_sum = log_gamma_sum + log_gamma

    result = log_gamma_sum + C
    result = tl.where(mask, result, 0.0)

    output_ptrs = output_ptr + offs_n * stride_on
    tl.store(output_ptrs, result, mask=mask)


def multigammaln(a: torch.Tensor, p: int):
    """Multivariate log-gamma function.

    Computes the log of the multivariate gamma function with dimension p:
    log(Γ_p(a)) = C + Σ_{i=1}^p log(Γ(a - (i-1)/2))

    where C = log(π) * p * (p-1) / 4
    """
    logger.debug("METAX GEMS MULTIGAMMALN")
    if a.numel() == 0:
        return torch.empty_like(a)

    # Ensure input is contiguous and float
    if not a.is_contiguous():
        a = a.contiguous()

    output = torch.empty_like(a)

    # Use 1D kernel for all cases
    grid = lambda META: (triton.cdiv(a.numel(), META["BLOCK_N"]),)
    with torch_device_fn.device(a.device):
        multigammaln_kernel_1d[grid](
            a,
            output,
            p,
            a.numel(),
            a.stride(0),
            output.stride(0),
            BLOCK_N=1024,
        )
    return output