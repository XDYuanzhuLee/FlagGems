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
def embedding_spdm_kernel(
    out_ptr,  # pointer to the output
    in_ptr,  # pointer to the input indices
    weight_ptr,  # pointer to the embedding weights
    N: tl.constexpr,  # embedding dimension
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel for embedding lookup with optimized memory access pattern."""
    pid = tle.program_id(0)
    out_ptr += pid * N
    in_ptr += pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)

    # Load the index for this position
    row_idx = tl.load(in_ptr)
    # Calculate the offset in the weight matrix
    weight_ptr += row_idx * N
    # Load the embedding vector
    embedding_weight = tl.load(weight_ptr + cols, mask, other=0.0)
    # Store the result
    tl.store(out_ptr + cols, embedding_weight, mask)


@libentry()
@triton.jit(do_not_specialize=["padding_idx"])
def embedding_spdm_backward_kernel(
    grad_in,  # pointer to the gradient input (embedding table gradients)
    grad_out,  # pointer to the gradient output (output gradients)
    indices,  # pointer to the input indices
    padding_idx,  # padding_idx to ignore
    HAS_PADDING_IDX: tl.constexpr,
    N: tl.constexpr,  # embedding dimension
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel for embedding backward with atomic add for gradient accumulation."""
    pid = tle.program_id(0)
    grad_out += pid * N
    indices += pid

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)

    row_idx = tl.load(indices).to(tl.int32)
    if not HAS_PADDING_IDX:
        grad_in += row_idx * N
        embedding_grad = tl.load(grad_out + cols, mask, other=0.0)
        if tl.constexpr(embedding_grad.dtype.is_bf16()):
            embedding_grad = embedding_grad.to(tl.float32)
        tl.atomic_add(grad_in + cols, embedding_grad, mask=mask)
    else:
        if row_idx != padding_idx:
            grad_in += row_idx * N
            embedding_grad = tl.load(grad_out + cols, mask, other=0.0)
            if tl.constexpr(embedding_grad.dtype.is_bf16()):
                embedding_grad = embedding_grad.to(tl.float32)
            tl.atomic_add(grad_in + cols, embedding_grad, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["n_rows"])
def embedding_spdm_grad_scale_kernel(
    grad_out,
    indice_freq,
    n_rows,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel to scale gradients by inverse frequency."""
    row_start = tle.program_id(0)
    row_step = tle.num_programs(0)

    for row_idx in range(row_start, n_rows, row_step):
        embedding_scale = 1.0
        indice_freq_val = tl.load(indice_freq + row_idx)
        if indice_freq_val > 1:
            embedding_scale = 1.0 / indice_freq_val

        cols = tl.arange(0, BLOCK_SIZE)
        mask = tl.arange(0, BLOCK_SIZE) < N
        embedding_grad = tl.load(grad_out + row_idx * N + cols, mask=mask)
        scaled_embedding_grad = embedding_grad * embedding_scale
        tl.store(grad_out + row_idx * N + cols, scaled_embedding_grad, mask=mask)


def embedding_spdm(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    """
    Embedding lookup operation for embeddingSpMDM.

    This is a specialized embedding kernel optimized for the Metax backend.

    Args:
        weight: Embedding table tensor of shape (num_embeddings, embedding_dim)
        indices: Indices tensor of shape (..., )
        padding_idx: Index to ignore (default: -1)
        scale_grad_by_freq: Whether to scale gradients by frequency (default: False)
        sparse: Whether to use sparse format (default: False, not supported)

    Returns:
        Embedding vectors of shape (..., embedding_dim)
    """
    logger.debug("METAX GEMS EMBEDDING SpMDM FORWARD")
    assert not sparse, "Currently do not support sparse format"

    M = indices.numel()
    N = weight.shape[-1]

    BLOCK_SIZE = triton.next_power_of_2(N)
    # TODO: remove contiguous enforcement
    indices = indices.contiguous()
    weight = weight.contiguous()
    output = torch.empty((*indices.shape, N), device=indices.device, dtype=weight.dtype)

    with torch_device_fn.device(weight.device):
        embedding_spdm_kernel[M,](output, indices, weight, N, BLOCK_SIZE)

    return output


def embedding_spdm_backward(
    grad_outputs,
    indices,
    num_weights,
    padding_idx=-1,
    scale_grad_by_freq=False,
    sparse=False,
):
    """
    Backward pass for embeddingSpMDM.

    Args:
        grad_outputs: Gradient of output of shape (..., embedding_dim)
        indices: Indices used in forward pass
        num_weights: Number of embeddings in the table
        padding_idx: Index to ignore (default: -1)
        scale_grad_by_freq: Whether to scale gradients by frequency (default: False)
        sparse: Whether to use sparse format (default: False, not supported)

    Returns:
        Gradient for embedding table of shape (num_weights, embedding_dim)
    """
    logger.debug("METAX GEMS EMBEDDING SpMDM BACKWARD")
    assert not sparse, "Currently do not support sparse format"

    M = indices.numel()
    N = grad_outputs.shape[-1]

    grad_inputs = torch.zeros(
        (num_weights, grad_outputs.shape[-1]),
        device=grad_outputs.device,
        dtype=(
            torch.float32
            if grad_outputs.dtype is torch.bfloat16
            else grad_outputs.dtype
        ),
    )

    if scale_grad_by_freq:
        indice_freq = torch.zeros(
            (num_weights,),
            requires_grad=False,
            device=grad_outputs.device,
            dtype=torch.int32,
        )
        INDICE_BLOCK_SIZE = 256
        indice_grid = (triton.cdiv(M, INDICE_BLOCK_SIZE),)

        with torch_device_fn.device(grad_outputs.device):
            # Reuse the indice_freq_kernel from the standard embedding
            from flag_gems.ops.embedding import indice_freq_kernel

            indice_freq_kernel[indice_grid](indice_freq, indices, M, INDICE_BLOCK_SIZE)
    else:
        indice_freq = None

    BLOCK_SIZE = triton.next_power_of_2(N)

    HAS_PADDING_IDX = padding_idx is not None

    with torch_device_fn.device(grad_outputs.device):
        embedding_spdm_backward_kernel[M,](
            grad_inputs,
            grad_outputs,
            indices,
            padding_idx,
            HAS_PADDING_IDX,
            N,
            BLOCK_SIZE,
        )

    if scale_grad_by_freq:
        with torch_device_fn.device(grad_outputs.device):
            embedding_spdm_grad_scale_kernel[M,](
                grad_inputs, indice_freq, num_weights, N, BLOCK_SIZE
            )
    return (
        grad_inputs.to(torch.bfloat16)
        if grad_outputs.dtype is torch.bfloat16
        else grad_inputs
    )