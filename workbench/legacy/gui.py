"""Compatibility alias used by migration-era GUI patches."""
import sys as _sys
from ..gui import main_window as _impl
_sys.modules[__name__] = _impl
