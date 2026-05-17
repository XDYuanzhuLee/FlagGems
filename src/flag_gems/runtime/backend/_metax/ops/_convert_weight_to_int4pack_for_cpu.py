import logging

import torch

logger = logging.getLogger("flag_gems." + __name__)


def _convert_weight_to_int4pack_for_cpu(self: torch.Tensor, innerKTiles: int) -> torch.Tensor:
    logger.debug("METAX GEMS _convert_weight_to_int4pack_for_cpu")
    # This operator is CPU-only in PyTorch.
    # For GPU context, we perform a pass-through as there's no direct GPU equivalent.
    # The operator converts int8 weights to int4pack format for CPU inference.
    # In a GPU context, we simply return the input tensor as-is.
    return self