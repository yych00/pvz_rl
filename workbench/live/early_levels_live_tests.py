"""Compatibility alias preserving monkey-patch identity."""
import sys as _sys
from . import early_levels as _impl
_sys.modules[__name__] = _impl
