"""Compatibility alias for the canonical 1-4 LIVE module."""
import sys as _sys
from . import level_1_4 as _impl
_sys.modules[__name__] = _impl
