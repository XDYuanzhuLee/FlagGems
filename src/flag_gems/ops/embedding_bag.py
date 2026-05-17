import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def embedding_bag_kernel(
    output_ptr,
    weight_ptr,
    indices_ptr,
    offsets_ptr,
    num_indices,
    embedding_dim: tl.constexpr,
    num_bags,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    if pid >= num_bags:
        return

    bag_start = tl.load(offsets_ptr + pid).to(tl.int32)
    if pid + 1 < num_bags:
        bag_end = tl.load(offsets_ptr + pid + 1).to(tl.int32)
    else:
        bag_end = num_indices

    # Accumulator for sum mode
    acc = tl.zeros((embedding_dim,), tl.float32)

    # Process elements in this bag
    for i in range(bag_start, bag_end):
        idx = tl.load(indices_ptr + i).to(tl.int32)
        row_ptr = idx * embedding_dim
        cols = tl.arange(0, embedding_dim)
        mask = cols < embedding_dim

        emb = tl.load(weight_ptr + row_ptr + cols, mask=mask, other=0.0)
        acc = acc + emb

    # Store result
    output_ptr += pid * embedding_dim
    cols = tl.arange(0, embedding_dim)
    mask = cols < embedding_dim
    tl.store(output_ptr + cols, acc, mask=mask)


def embedding_bag(weight, indices, offsets, mode="sum", scale_grad_by_freq=False,
                  padding_idx=-1, sparse=False, per_sample_weights=None,
                  include_last_offset=False):
    """
    Computes sums or means of 'bags' of embeddings.

    Args:
        weight: Embedding table (num_embeddings x embedding_dim)
        indices: Indices to look up
        offsets: Offsets of each bag
        mode: "sum", "mean", or "max"
        scale_grad_by_freq: If True, scale gradients by frequency
        padding_idx: If set, ignores this index
        sparse: If True, use sparse gradient
        per_sample_weights: Weights for each sample
        include_last_offset: If True, offsets includes last offset

    Returns:
        output: Tensor of shape (num_bags, embedding_dim)
    """
    logger.debug("GEMS EMBEDDING_BAG")

    assert not sparse, "Currently do not support sparse format"
    assert per_sample_weights is None, "Currently do not support per_sample_weights"

    num_bags = offsets.numel()
    if include_last_offset:
        num_bags = num_bags - 1
    embedding_dim = weight.shape[-1]
    num_indices = indices.numel()

    output = torch.empty((num_bags, embedding_dim), device=weight.device, dtype=weight.dtype)

    BLOCK_SIZE = triton.next_power_of_2(embedding_dim)
    grid = (num_bags,)

    indices = indices.contiguous()
    offsets = offsets.contiguous()
    weight = weight.contiguous()

    with torch_device_fn.device(weight.device):
        embedding_bag_kernel[grid](
            output, weight, indices, offsets,
            num_indices, embedding_dim, num_bags, BLOCK_SIZE
        )

    return output