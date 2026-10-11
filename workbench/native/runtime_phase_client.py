# -*- coding: utf-8 -*-
"""Python binding patch for the generic native runtime-phase API.

`pvz_gym_get_ui()` is kept as the raw original LawnApp::mGameScene value for
low-level diagnostics. Higher-level orchestration should use
`get_runtime_state()` / `get_runtime_phase()` instead of interpreting raw scene
numbers directly.
"""

from __future__ import annotations

import ctypes
import json
from typing import Any, Dict, Tuple


RAW_SCENE_MAP = {
    -1: "未就绪/未连接 (-1)",
    0: "SCENE_LOADING (0)",
    1: "SCENE_MENU (1)",
    2: "SCENE_LEVEL_INTRO (2)",
    3: "SCENE_PLAYING (3)",
    4: "SCENE_ZOMBIES_WON / 玩家失败 (4)",
    5: "SCENE_AWARD / 通关奖励 (5)",
    6: "SCENE_CREDIT (6)",
    7: "SCENE_CHALLENGE (7)",
}

PHASE_MAP = {
    0: "UNKNOWN",
    1: "OFFLINE",
    2: "LOADING",
    3: "MENU",
    4: "LEVEL_INTRO",
    5: "DIALOGUE",
    6: "SEED_SELECTION",
    7: "BATTLE",
    8: "LOST",
    9: "AWARD",
    10: "CREDIT",
    11: "CHALLENGE",
    12: "LEVEL_COMPLETE",
}


def install_runtime_phase_client(client_cls: Any) -> None:
    """Install runtime phase bindings before any PvzGymNativeDll instance exists."""
    if getattr(client_cls, "_runtime_phase_client_installed", False):
        return

    original_bind = client_cls._bind_functions

    def _bind_functions_with_runtime_phase(self):
        original_bind(self)
        dll = getattr(self, "dll", None)
        if dll is None:
            return

        if hasattr(dll, "pvz_gym_get_runtime_phase"):
            dll.pvz_gym_get_runtime_phase.argtypes = []
            dll.pvz_gym_get_runtime_phase.restype = ctypes.c_int

        if hasattr(dll, "pvz_gym_get_runtime_state_json"):
            dll.pvz_gym_get_runtime_state_json.argtypes = [ctypes.c_char_p, ctypes.c_int]
            dll.pvz_gym_get_runtime_state_json.restype = ctypes.c_int

    def get_runtime_phase(self) -> Tuple[int, str]:
        dll = getattr(self, "dll", None)
        if dll is None or not hasattr(dll, "pvz_gym_get_runtime_phase"):
            return 0, "UNKNOWN"
        try:
            value = int(dll.pvz_gym_get_runtime_phase())
            return value, PHASE_MAP.get(value, f"UNKNOWN({value})")
        except Exception:
            return 0, "UNKNOWN"

    def get_runtime_state(self, buf_size: int = 4096) -> Dict[str, Any]:
        dll = getattr(self, "dll", None)
        if dll is None or not hasattr(dll, "pvz_gym_get_runtime_state_json"):
            return {
                "process_available": False,
                "raw_scene": -1,
                "raw_scene_name": "UNAVAILABLE",
                "phase": 0,
                "phase_name": "UNKNOWN",
                "game_status": 0,
                "board_present": False,
                "cutscene_present": False,
                "seed_choosing": False,
                "dave_talking": False,
                "level_complete": False,
                "game_mode": -1,
                "level_id": 0,
                "finished_adventure": -1,
                "error": "runtime phase API unavailable; rebuild pvz_gym_native.dll",
            }

        try:
            buf = ctypes.create_string_buffer(buf_size)
            written = int(dll.pvz_gym_get_runtime_state_json(buf, buf_size))
            if written <= 0:
                return {
                    "phase": 0,
                    "phase_name": "UNKNOWN",
                    "finished_adventure": -1,
                    "error": f"runtime state read failed: {written}",
                }
            state = json.loads(buf.value.decode("utf-8", errors="replace"))
            state.setdefault("finished_adventure", -1)
            return state
        except Exception as exc:
            return {
                "phase": 0,
                "phase_name": "UNKNOWN",
                "finished_adventure": -1,
                "error": str(exc),
            }

    client_cls._bind_functions = _bind_functions_with_runtime_phase
    client_cls.get_runtime_phase = get_runtime_phase
    client_cls.get_runtime_state = get_runtime_state
    client_cls.UI_MAP = dict(RAW_SCENE_MAP)
    client_cls.RUNTIME_PHASE_MAP = dict(PHASE_MAP)
    client_cls._runtime_phase_client_installed = True
