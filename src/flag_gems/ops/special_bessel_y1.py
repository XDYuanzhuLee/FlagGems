import torch

import flag_gems


def special_bessel_y1(x: torch.Tensor):
    """
    ATen wrapper: special_bessel_y1(Tensor self) -> Tensor
    """
    return torch.ops.aten.special_bessel_y1(x)


def special_bessel_y1_out(x: torch.Tensor, out: torch.Tensor):
    """
    ATen wrapper: special_bessel_y1.out(Tensor self, Tensor out) -> Tensor
    """
    out.copy_(torch.ops.aten.special_bessel_y1(x))
    return out