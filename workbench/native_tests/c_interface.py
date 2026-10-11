"""Compatibility alias for native test modules kept unchanged during move."""
import sys as _sys
from ..native import c_interface as _impl
_sys.modules[__name__] = _impl
