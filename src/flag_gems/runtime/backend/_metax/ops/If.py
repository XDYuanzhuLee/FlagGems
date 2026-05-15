import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger("flag_gems." + __name__)


@pointwise_dynamic(
    is_tensor=[True, True, True],
    promotion_methods=[(1, 2, "NO_OPMATH")],
)
@triton.jit
def if_inner(condition, true_val, false_val):
    return tl.where(condition, true_val, false_val)


def if_function(condition, true_val, false_val):
    """Conditional select: returns true_val where condition is True, else false_val.

    METAX specialized implementation using Triton kernels.
    """
    logger.debug("METAX GEMS IF")

    # Handle scalar inputs
    if not isinstance(condition, torch.Tensor):
        condition = torch.tensor(condition, device=true_val.device)
    if not isinstance(true_val, torch.Tensor):
        true_val = torch.tensor(true_val, device=condition.device)
    if not isinstance(false_val, torch.Tensor):
        false_val = torch.tensor(false_val, device=condition.device)

    # Get the result type
    result_type = torch.result_type(true_val, false_val)

    # Convert tensors to the result type if needed
    if true_val.dtype != result_type:
        true_val = true_val.to(result_type)
    if false_val.dtype != result_type:
        false_val = false_val.to(result_type)

    # Get the device
    devices = [condition.device, true_val.device, false_val.device]
    devices = [d for d in devices if d.type != "cpu"]

    assert len(devices), "CPU only. There seems a mistake to dispatch to here."
    device = devices[0]

    with torch_device_fn.device(device):
        # Ensure all tensors are on the same device
        if condition.device != device and condition.ndim == 0:
            condition = condition.to(device)
        if true_val.device != device and true_val.ndim == 0:
            true_val = true_val.to(device)
        if false_val.device != device and false_val.ndim == 0:
            false_val = false_val.to(device)

        # Ensure condition is boolean
        if condition.dtype != torch.bool:
            condition = condition.to(torch.bool)

        # Broadcast all tensors to the same shape
        out_shape = torch.broadcast_shapes(condition.shape, true_val.shape, false_val.shape)
        out = torch.empty(out_shape, dtype=result_type, device=device)

        ndim = max(condition.ndim, true_val.ndim, false_val.ndim)
        if_inner.instantiate(ndim)
        if_inner(condition, true_val, false_val, out0=out)

    return out