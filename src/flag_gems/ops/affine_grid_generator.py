import torch
import torch.nn.functional as F


def affine_grid_generator(theta, size, align_corners=False):
    """
    Generates a 2D sampling grid from an affine transformation matrix.

    This is a wrapper around torch.affine_grid_generator that can be replaced
    by backend-specific implementations.

    Args:
        theta: (N, 2, 3) tensor of affine transformation matrices
        size: List of 4 integers [N, C, H, W]
        align_corners: If True, the corner pixels of the input and output
                      are aligned.

    Returns:
        grid: (N, H, W, 2) tensor of normalized coordinates
    """
    return torch.nn.functional.affine_grid_generator(theta, size, align_corners)