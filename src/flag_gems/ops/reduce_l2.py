import torch

from flag_gems.ops.vector_norm import vector_norm


def reduce_l2(inp, dim=None, keepdim=False):
    """Compute the L2 norm (Euclidean norm) of the input tensor.

    This is equivalent to torch.linalg.vector_norm(inp, ord=2, dim=dim, keepdim=keepdim)
    """
    return vector_norm(inp, ord=2, dim=dim, keepdim=keepdim)