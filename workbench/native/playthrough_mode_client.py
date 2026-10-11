# -*- coding: utf-8 -*-
"""Bind the native single-level Adventure playthrough override.

The project replays one selected stage in isolation.  Therefore "一周目/二周目"
is an episode configuration rather than a persistent save-progression decision.
"""

from __future__ import annotations

import ctypes
from typing import Any, Tuple


def install_playthrough_mode_client(client_cls: Any) -> None:
    if getattr(client_cls, "_playthrough_mode_client_installed", False):
        return

    original_bind = client_cls._bind_functions

    def _bind_functions_with_playthrough(self):
        original_bind(self)
        dll = getattr(self, "dll", None)
        if dll is None:
            return
        if hasattr(dll, "pvz_gym_set_adventure_playthrough"):
            dll.pvz_gym_set_adventure_playthrough.argtypes = [ctypes.c_int]
            dll.pvz_gym_set_adventure_playthrough.restype = ctypes.c_int

    def set_adventure_playthrough(self, playthrough: int) -> Tuple[int, str]:
        """Force this process to emulate Adventure pass 1 or 2 for the next episode."""
        try:
            playthrough = int(playthrough)
        except Exception:
            return 1002, "invalid playthrough"
        if playthrough not in (1, 2):
            return 1002, "playthrough must be 1 or 2"

        dll = getattr(self, "dll", None)
        if dll is None:
            return -1, "DLL 未加载"
        if not hasattr(dll, "pvz_gym_set_adventure_playthrough"):
            return -1, "DLL 缺少 pvz_gym_set_adventure_playthrough；请重新编译 native DLL"

        try:
            code = int(dll.pvz_gym_set_adventure_playthrough(playthrough))
            if hasattr(self, "error_message"):
                return code, self.error_message(code)
            return code, "success" if code == 0 else f"error {code}"
        except Exception as exc:
            return -1, f"调用异常: {exc}"

    client_cls._bind_functions = _bind_functions_with_playthrough
    client_cls.set_adventure_playthrough = set_adventure_playthrough
    client_cls._playthrough_mode_client_installed = True
