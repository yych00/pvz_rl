"""Compatibility alias for the LIVE level registry."""
import sys as _sys
from ..live import registry as _impl
_sys.modules[__name__] = _impl
