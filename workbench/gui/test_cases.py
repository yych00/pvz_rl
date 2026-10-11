"""Compatibility alias for native workbench test cases."""
import sys as _sys
from ..native_tests import cases as _impl
_sys.modules[__name__] = _impl
