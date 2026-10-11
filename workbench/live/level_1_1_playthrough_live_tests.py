"""Compatibility alias for the canonical 1-1 playthrough module."""
import sys as _sys
from . import level_1_1_playthrough as _impl
_sys.modules[__name__] = _impl
