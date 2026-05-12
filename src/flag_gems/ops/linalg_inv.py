import logging

import torch

logger = logging.getLogger(__name__)


def linalg_inv(A):
    return torch.linalg.inv(A)