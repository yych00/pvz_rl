"""Native bridge for the seed-selection action domain."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

from .native import NativePvzBackend, NativeCapabilityError, _call_with_resizable_buffer
from ..seed_types import SeedOptionState, SeedSelectionResult, SeedSelectionState
from ..seed_types import SEED_ACTION_COUNT, SEED_TYPE_CAPACITY


class SeedSelectionNativeBackend(NativePvzBackend):
    """Extends the shared native backend with SeedChooser-only capabilities."""

    def __init__(self, dll_path: str | Path | None = None):
        super().__init__(dll_path)
        d = self._dll
        required = [
            "pvz_gym_get_seed_type_capacity",
            "pvz_gym_get_seed_selection_json",
            "pvz_gym_get_seed_selection_mask",
            "pvz_gym_select_seed",
            "pvz_gym_select_imitater",
            "pvz_gym_deselect_seed",
            "pvz_gym_confirm_seeds",
        ]
        missing = [name for name in required if not hasattr(d, name)]
        if missing:
            raise NativeCapabilityError(f"seed-selection ABI missing from DLL: {missing}")

        d.pvz_gym_get_seed_type_capacity.argtypes = []
        d.pvz_gym_get_seed_type_capacity.restype = ctypes.c_int
        d.pvz_gym_get_seed_selection_json.argtypes = [ctypes.c_char_p, ctypes.c_int]
        d.pvz_gym_get_seed_selection_json.restype = ctypes.c_int
        d.pvz_gym_get_seed_selection_mask.argtypes = [
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.POINTER(ctypes.c_int)
        ]
        d.pvz_gym_get_seed_selection_mask.restype = ctypes.c_int
        d.pvz_gym_select_seed.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        d.pvz_gym_select_seed.restype = ctypes.c_int
        d.pvz_gym_select_imitater.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        d.pvz_gym_select_imitater.restype = ctypes.c_int
        d.pvz_gym_deselect_seed.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        d.pvz_gym_deselect_seed.restype = ctypes.c_int
        d.pvz_gym_confirm_seeds.argtypes = [ctypes.c_char_p, ctypes.c_int]
        d.pvz_gym_confirm_seeds.restype = ctypes.c_int

    def _ensure_attached(self) -> None:
        if not self._dll.pvz_gym_attach():
            raise NativeCapabilityError("PlantsVsZombies.exe is not attached")

    def get_seed_type_capacity(self) -> int:
        capacity = int(self._dll.pvz_gym_get_seed_type_capacity())
        if capacity != SEED_TYPE_CAPACITY:
            raise NativeCapabilityError(f"seed codec requires capacity {SEED_TYPE_CAPACITY}, got {capacity}")
        return capacity

    def get_seed_selection(self) -> SeedSelectionState:
        self._ensure_attached()
        ret, payload = _call_with_resizable_buffer(
            lambda buf, size: self._dll.pvz_gym_get_seed_selection_json(buf, size),
            initial_size=8192,
        )
        if ret < 0:
            raise NativeCapabilityError(f"pvz_gym_get_seed_selection_json returned {ret}")
        if ret == 0 or not payload:
            return SeedSelectionState()

        data = json.loads(payload.decode("utf-8", errors="replace"))
        options = tuple(
            SeedOptionState(
                plant_type=int(item.get("plant_type", -1)),
                available=bool(item.get("available", False)),
                unlocked=bool(item.get("unlocked", False)),
                selected=bool(item.get("selected", False)),
                disabled=bool(item.get("disabled", False)),
                chooser_state=int(item.get("chooser_state", 0)),
            )
            for item in data.get("options", [])
        )
        return SeedSelectionState(
            present=bool(data.get("present", False)),
            required_count=int(data.get("required_count", 0)),
            selected_count=int(data.get("selected_count", 0)),
            can_confirm=bool(data.get("can_confirm", False)),
            seeds_in_flight=bool(data.get("seeds_in_flight", False)),
            warning_flags=int(data.get("warning_flags", 0)),
            imitater_target=int(data.get("imitater_target", -1)),
            selected_seeds=tuple(int(v) for v in data.get("selected_seeds", [])),
            options=options,
            zombie_types_in_level=(tuple(int(v) for v in data.get("zombie_types_in_level", []))
                                   if data.get("zombie_roster_available", False) else None),
        )

    def get_seed_selection_mask(self) -> list[bool]:
        self._ensure_attached()
        capacity = self.get_seed_type_capacity()
        expected = SEED_ACTION_COUNT
        buf = (ctypes.c_uint8 * expected)()
        out_count = ctypes.c_int(0)
        ret = int(self._dll.pvz_gym_get_seed_selection_mask(buf, expected, ctypes.byref(out_count)))
        if ret < 0:
            raise NativeCapabilityError(f"pvz_gym_get_seed_selection_mask returned {ret}")
        if out_count.value != expected:
            raise NativeCapabilityError(f"seed action mask length {out_count.value}, expected {expected}")
        return [bool(buf[i]) for i in range(expected)]

    @staticmethod
    def _result(code: int, buf: ctypes.Array) -> SeedSelectionResult:
        message = buf.value.decode("utf-8", errors="replace")
        return SeedSelectionResult(success=(code == 0), error_code=code, message=message)

    def select_seed(self, plant_type: int) -> SeedSelectionResult:
        self._ensure_attached()
        buf = ctypes.create_string_buffer(512)
        code = int(self._dll.pvz_gym_select_seed(int(plant_type), buf, len(buf)))
        return self._result(code, buf)

    def select_imitater(self, target_type: int) -> SeedSelectionResult:
        self._ensure_attached()
        buf = ctypes.create_string_buffer(512)
        code = int(self._dll.pvz_gym_select_imitater(int(target_type), buf, len(buf)))
        return self._result(code, buf)

    def deselect_seed(self, plant_type: int) -> SeedSelectionResult:
        self._ensure_attached()
        buf = ctypes.create_string_buffer(512)
        code = int(self._dll.pvz_gym_deselect_seed(int(plant_type), buf, len(buf)))
        return self._result(code, buf)

    def confirm_seeds(self) -> SeedSelectionResult:
        self._ensure_attached()
        buf = ctypes.create_string_buffer(512)
        code = int(self._dll.pvz_gym_confirm_seeds(buf, len(buf)))
        return self._result(code, buf)
