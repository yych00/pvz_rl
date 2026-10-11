"""Compatibility alias for the canonical 1-1 LIVE module."""
import sys as _sys
from . import level_1_1 as _impl
_sys.modules[__name__] = _impl
