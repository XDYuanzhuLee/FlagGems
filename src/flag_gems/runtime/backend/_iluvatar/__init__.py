from flag_gems.runtime.backend.backend_utils import VendorInfoBase

vendor_info = VendorInfoBase(
    vendor_name="iluvatar", device_name="cuda", device_query_cmd="ixsmi"
)

ARCH_MAP = {
    "7": "iluvatar",
}

CUSTOMIZED_UNUSED_OPS = ()

__all__ = ["*"]
