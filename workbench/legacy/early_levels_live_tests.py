"""Compatibility alias preserving the original LIVE monkey-patch target."""
import sys as _sys
from ..live import early_levels as _impl
_sys.modules[__name__] = _impl
