from .addmm import addmm
from .amax import amax
from .arange import arange, arange_start
from .bmm import bmm
from .exponential_ import exponential_
from .full import full
from .full_like import full_like
from .groupnorm import group_norm
from .index import index
from .index_put import index_put, index_put_
from .index_select import index_select
from .isin import isin
from .log_softmax import log_softmax, log_softmax_backward
from .masked_fill import masked_fill, masked_fill_
from .min import min, min_dim
from .mm import mm, mm_out
from .nonzero import nonzero
from .ones import ones
from .ones_like import ones_like
from .outer import outer
from .polar import polar
from .prod import prod, prod_dim
from .repeat_interleave import repeat_interleave_self_tensor
from .resolve_conj import resolve_conj
from .sigmoid import sigmoid
from .tanh import tanh
from .unique import _unique2
from .upsample_nearest2d import upsample_nearest2d
from .zeros import zeros
from .zeros_like import zeros_like

# Import scaled_dot_product_cudnn_attention_backward with error handling
try:
    from .scaled_dot_product_cudnn_attention_backward import (
        scaled_dot_product_cudnn_attention_backward,
    )
except ImportError:
    # Fallback: try to import from absolute path
    import importlib.util
    import os
    import sys

    ops_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(ops_dir, '_scaled_dot_product_cudnn_attention_backward.py')

    if os.path.exists(module_path):
        spec = importlib.util.spec_from_file_location(
            'scaled_dot_product_cudnn_attention_backward',
            module_path
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules['scaled_dot_product_cudnn_attention_backward'] = module
            spec.loader.exec_module(module)
            scaled_dot_product_cudnn_attention_backward = module.scaled_dot_product_cudnn_attention_backward
    else:
        scaled_dot_product_cudnn_attention_backward = None

__all__ = [
    "_unique2",
    "addmm",
    "amax",
    "arange",
    "arange_start",
    "bmm",
    "exponential_",
    "full",
    "full_like",
    "group_norm",
    "index",
    "index_put",
    "index_put_",
    "index_select",
    "isin",
    "log_softmax",
    "log_softmax_backward",
    "masked_fill",
    "masked_fill_",
    "min_dim",
    "min",
    "mm",
    "mm_out",
    "nonzero",
    "ones",
    "ones_like",
    "outer",
    "polar",
    "prod",
    "prod_dim",
    "repeat_interleave_self_tensor",
    "resolve_conj",
    "scaled_dot_product_cudnn_attention_backward",
    "sigmoid",
    "tanh",
    "upsample_nearest2d",
    "zeros",
    "zeros_like",
]
