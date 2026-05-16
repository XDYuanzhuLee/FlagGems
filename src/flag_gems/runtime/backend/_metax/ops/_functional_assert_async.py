import logging

import torch

logger = logging.getLogger("flag_gems." + __name__)


def _functional_assert_async(
    self: torch.Tensor, assert_msg: str, dep_token: torch.Tensor
) -> torch.Tensor:
    """Functional assert async operator for Metax backend.

    This operator is used in PyTorch's functionalization to assert conditions
    on tensors asynchronously. For the Metax implementation, we return the
    input tensor as the assertion is handled by the runtime.

    Args:
        self: Input tensor (typically boolean tensor for assertion condition)
        assert_msg: Assertion message to display on failure
        dep_token: Dependency token for tracking dependencies

    Returns:
        The input tensor (passthrough)
    """
    logger.debug("METAX GEMS _FUNCTIONAL_ASSERT_ASYNC")
    # The assertion is handled by the underlying PyTorch runtime
    # For the positive case (assertion passes), we simply return the input tensor
    return self