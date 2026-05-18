from backend_utils import VendorInfoBase

vendor_info = VendorInfoBase(
    vendor_name="metax", device_name="cuda", device_query_cmd="mx-smi"
)

CUSTOMIZED_UNUSED_OPS = ()


def _patch_ops():
    """Patch flag_gems.ops with metax specialized implementations.

    This is called after flag_gems is fully initialized to avoid circular imports.
    """
    import flag_gems
    import flag_gems.ops

    # Only patch if flag_gems.ops is available and has the original function
    if hasattr(flag_gems, 'ops') and hasattr(flag_gems.ops, 'scaled_dot_product_attention'):
        from flag_gems.runtime.backend._metax.ops import scaled_dot_product_attention
        # Replace scaled_dot_product_attention in flag_gems.ops
        flag_gems.ops.scaled_dot_product_attention = scaled_dot_product_attention


# Register the patch to be called after flag_gems is initialized
# We do this by adding a finalizer
import atexit


def _apply_delayed_patch():
    try:
        import flag_gems
        if hasattr(flag_gems, 'ops'):
            _patch_ops()
    except Exception:
        pass


# Try to apply the patch now if flag_gems is already imported
try:
    import flag_gems
    if hasattr(flag_gems, 'ops'):
        _patch_ops()
except Exception:
    pass

__all__ = ["vendor_info"]
