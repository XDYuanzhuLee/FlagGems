import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger("flag_gems." + __name__)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def log_sigmoid_forward(x):
    return tl.minimum(x, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(x).to(tl.float32)))


def log_sigmoid(x):
    logger.debug("METAX GEMS LOG_SIGMOID FORWARD")

    return log_sigmoid_forward(x)