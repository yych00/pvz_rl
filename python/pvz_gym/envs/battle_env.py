"""Backward-compatible alias for :mod:`pvz_gym.envs.battle`."""
import sys as _sys
from . import battle as _impl
_sys.modules[__name__] = _impl
