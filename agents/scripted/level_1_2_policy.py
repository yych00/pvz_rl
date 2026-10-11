"""Backward-compatible module alias for :mod:`agents.scripted.level_1_2`."""
import sys as _sys
from . import level_1_2 as _impl
_sys.modules[__name__] = _impl
