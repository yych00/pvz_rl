"""Compatibility alias for the reorganized LIVE engine."""
import sys as _sys
from ..live import early_levels as _impl
_sys.modules[__name__] = _impl
