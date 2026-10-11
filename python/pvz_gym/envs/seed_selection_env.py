"""Backward-compatible alias for :mod:`pvz_gym.envs.seed_selection`."""
import sys as _sys
from . import seed_selection as _impl
_sys.modules[__name__] = _impl
