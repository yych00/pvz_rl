"""Compatibility alias used by migration-era patches."""
import sys as _sys
from ..native import c_interface as _impl
_sys.modules[__name__] = _impl
