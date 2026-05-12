import torch


def linalg_vander(x, N=None):
    # Handle N parameter
    if N is None:
        N = x.shape[-1]

    # Get input shape info
    batch_dims = x.shape[:-1]
    n = x.shape[-1]

    # Flatten batch dims
    x_flat = x.reshape(-1)

    # Compute Vandermonde matrix
    # For each input element x[i], output [1, x[i], x[i]^2, ..., x[i]^(N-1)]
    powers = torch.arange(N, device=x.device, dtype=torch.float32)
    output = torch.pow(x_flat.unsqueeze(1), powers)

    # Reshape output to final shape (*, n, N)
    final_shape = batch_dims + (n, N)
    output = output.reshape(final_shape)

    # Convert back to original dtype
    output = output.to(x.dtype)

    return output