import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

logger = logging.getLogger("flag_gems." + __name__)


@triton.jit
def moe_gate_topk_routing_kernel(
    gate_logits_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    num_tokens,
    num_experts,
    k,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_EXPERTS: tl.constexpr,
):
    """Triton kernel for MoE Gate Top-K Routing.

    Computes:
    1. Softmax over gate_logits along the expert dimension
    2. Top-K selection of experts
    3. Normalization of top-k weights
    """
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    valid_rows = rows < num_tokens

    # Load gate logits
    cols = tl.arange(0, BLOCK_SIZE_EXPERTS)
    valid_cols = cols < num_experts

    logits = tl.load(
        gate_logits_ptr + rows[:, None] * num_experts + cols[None, :],
        mask=valid_rows[:, None] & valid_cols[None, :],
        other=-float("inf"),
    ).to(tl.float32)

    # Compute softmax with numerical stability
    row_max = tl.max(logits, axis=1)[:, None]
    exp_vals = tl.exp(logits - row_max)
    probs = exp_vals / (tl.sum(exp_vals, axis=1)[:, None] + 1e-8)

    # Extract top-K experts
    selected_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for ki in range(k):
        curr_max, curr_arg = tl.max(probs, axis=1, return_indices=True)

        tl.store(topk_weights_ptr + rows * k + ki, curr_max, mask=valid_rows)
        tl.store(topk_ids_ptr + rows * k + ki, curr_arg, mask=valid_rows)
        selected_sum += curr_max

        # Mask out selected expert for next iteration
        probs = tl.where(
            cols[None, :] == curr_arg[:, None], -float("inf"), probs
        )

    # Normalize weights
    norm = selected_sum + 1e-8
    for ki in range(k):
        idx = rows * k + ki
        val = tl.load(topk_weights_ptr + idx, mask=valid_rows)
        tl.store(topk_weights_ptr + idx, val / norm, mask=valid_rows)


def moe_gate_topk_routing(gate_logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """MoE Gate Top-K Routing.

    Applies softmax to gate logits and selects top-k experts.

    Args:
        gate_logits: (M, E) - raw gate values for each expert
        top_k: Number of top experts to select

    Returns:
        topk_weights: (M, K) - normalized weights for top-k experts
        topk_ids: (M, K) - indices of top-k experts
    """
    logger.debug("METAX GEMS MoE_Gate_TopK_Routing")

    num_tokens, num_experts = gate_logits.shape
    k = top_k

    # Allocate output tensors
    topk_weights = torch.empty((num_tokens, k), dtype=gate_logits.dtype, device=gate_logits.device)
    topk_ids = torch.empty((num_tokens, k), dtype=torch.int64, device=gate_logits.device)

    # Determine block sizes
    max_total_threads = 1024
    BLOCK_SIZE_EXPERTS = ((triton.next_power_of_2(num_experts) + 31) // 32) * 32
    BLOCK_SIZE_EXPERTS = min(BLOCK_SIZE_EXPERTS, 1024)
    BLOCK_SIZE = max_total_threads // BLOCK_SIZE_EXPERTS
    BLOCK_SIZE = max(BLOCK_SIZE, 1)

    # If num_experts > 128, use single row per thread block
    if num_experts > 128:
        BLOCK_SIZE = 1

    # Launch kernel
    grid = (triton.cdiv(num_tokens, BLOCK_SIZE),)

    moe_gate_topk_routing_kernel[grid](
        gate_logits,
        topk_weights,
        topk_ids,
        num_tokens,
        num_experts,
        k,
        BLOCK_SIZE,
        BLOCK_SIZE_EXPERTS,
    )

    return topk_weights, topk_ids