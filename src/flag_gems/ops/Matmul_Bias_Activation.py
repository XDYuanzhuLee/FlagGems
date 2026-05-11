import torch
import torch.nn.functional as F


def matmul_bias_activation(input, weight, bias):
    """
    Fused matmul + bias + ReLU activation.

    This is a generic implementation that calls torch operations.

    Args:
        input: Input tensor of shape (..., K) or (M, K)
        weight: Weight tensor of shape (N, K)
        bias: Bias tensor of shape (N,)

    Returns:
        Output tensor of shape (..., N) or (M, N)
    """
    # Compute matmul: input @ weight.T
    output = torch.matmul(input, weight.t())
    # Add bias
    output = output + bias
    # Apply ReLU activation
    output = F.relu(output)
    return output