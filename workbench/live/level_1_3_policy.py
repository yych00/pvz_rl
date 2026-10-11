"""Compatibility alias for the scripted 1-3 policy."""
import sys as _sys
from agents.scripted import level_1_3 as _impl
_sys.modules[__name__] = _impl
