"""Compatibility alias for 1-1 playthrough LIVE helpers."""
import sys as _sys
from ..live import level_1_1_playthrough as _impl
_sys.modules[__name__] = _impl
