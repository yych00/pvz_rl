"""High-level native client configuration and compatibility helpers."""

from __future__ import annotations

from .profiling import bridge_timing

import atexit
import ctypes
import json
from collections.abc import Mapping
from pathlib import Path

from .action_space import ACTION_SPACE_SIZE
from .bridge.native import (
    NativeCapabilityError,
    _action_ints,
    _call_with_resizable_buffer,
    _parse_step_json,
)
from .bridge.seed_selection import SeedSelectionNativeBackend
from .save_cleaner import clean_adventure_save
from .runtime import (TurboConfig, RuntimeConfig, RuntimePhase, PhaseAdvanceRequest,
    PhaseAdvanceResult, RuntimeSnapshot, SnapshotIdentity, coerce_config)
from .bridge.wire import decode_payload
from .bridge.native import _parse_observation_data
from .types import Action, ActionResult, GameStatus, Observation, ResetFacts, StepFacts


class PvZClient(SeedSelectionNativeBackend):
    """Thin configured transport over the authoritative native session API.

    Episode lifecycle and battle transitions live in native C++; this class only
    applies user-facing automation defaults and retains compatibility helpers used
    by the workbench/older callers.
    """

    def __init__(self, dll_path: str | Path | None = None, *, automation_enabled: bool = True,
                 auto_dialogue: bool = True, auto_collect: bool = True,
                 auto_whack: bool = True,
                 auto_select_seeds: bool = True, auto_skip_tutorial: bool = True,
                 background_running: bool = True, deterministic_pause: bool = True,
                 runtime_config: RuntimeConfig | Mapping | None = None,
                 save_dir: str | Path | None = None, clean_save: bool | None = None,
                 profile_id: int | None = None, instance_id: str | None = None,
                 template_dir: str | Path | None = None):
        super().__init__(dll_path)
        self.runtime_config = coerce_config(RuntimeConfig, runtime_config)
        self.configure_runtime(self.runtime_config)
        self.save_dir = Path(save_dir).resolve() if save_dir else None
        self.clean_save = bool(self.save_dir) if clean_save is None else bool(clean_save)
        self.profile_id = profile_id
        self.instance_id = instance_id
        self.template_dir = template_dir
        if bridge_timing.enabled:
            profiling = getattr(self._dll, "pvz_gym_set_step_profiling", None)
            if profiling is not None:
                profiling.argtypes = [ctypes.c_bool]
                profiling.restype = None
                profiling(True)
        self.auto_select_seeds = bool(auto_select_seeds)
        self.auto_skip_tutorial = bool(auto_skip_tutorial)
        self.background_running = bool(background_running)
        self.deterministic_pause = bool(deterministic_pause)
        self._turbo_active = False
        self._declare_client_signatures()
        self._old_automation = self.get_automation_config()
        self._old_auto_skip_tutorial = bool(self._dll.pvz_gym_get_auto_skip_tutorial())
        self.set_automation_config(automation_enabled, auto_dialogue, auto_collect)
        self.set_auto_whack(auto_whack)
        self._dll.pvz_gym_set_auto_skip_tutorial(self.auto_skip_tutorial)
        self._dll.pvz_gym_set_background_running(self.background_running)
        atexit.register(self.close)

    def _declare_client_signatures(self) -> None:
        d = self._dll
        d.pvz_gym_set_auto_skip_tutorial.argtypes = [ctypes.c_bool]
        d.pvz_gym_set_auto_skip_tutorial.restype = None
        d.pvz_gym_get_auto_skip_tutorial.argtypes = []
        d.pvz_gym_get_auto_skip_tutorial.restype = ctypes.c_bool
        d.pvz_gym_skip_tutorial.argtypes = []
        d.pvz_gym_skip_tutorial.restype = ctypes.c_bool
        d.pvz_gym_set_background_running.argtypes = [ctypes.c_bool]
        d.pvz_gym_set_background_running.restype = None
        d.pvz_gym_get_dialog.argtypes = [ctypes.c_int]
        d.pvz_gym_get_dialog.restype = ctypes.c_uint32
        d.pvz_gym_kill_dialog.argtypes = [ctypes.c_int]
        d.pvz_gym_kill_dialog.restype = ctypes.c_bool
        d.pvz_gym_begin_headless_mode.argtypes = []
        d.pvz_gym_begin_headless_mode.restype = ctypes.c_bool
        d.pvz_gym_advance_ticks_headless.argtypes = [ctypes.c_int]
        d.pvz_gym_advance_ticks_headless.restype = ctypes.c_int
        d.pvz_gym_end_headless_mode.argtypes = []
        d.pvz_gym_end_headless_mode.restype = ctypes.c_bool
        headless_status = getattr(d, "pvz_gym_get_headless_begin_status", None)
        if headless_status is not None:
            headless_status.argtypes = []
            headless_status.restype = ctypes.c_int
        headless_diag = getattr(d, "pvz_gym_get_headless_diagnostics_json", None)
        if headless_diag is not None:
            headless_diag.argtypes = [ctypes.c_char_p, ctypes.c_int]
            headless_diag.restype = ctypes.c_int
        action_args = [ctypes.c_int] * 7
        d.pvz_gym_env_step_headless.argtypes = action_args + [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int
        ]
        d.pvz_gym_env_step_headless.restype = ctypes.c_int
        continuous_step = getattr(d, "pvz_gym_env_step_continuous", None)
        if continuous_step is not None:
            continuous_step.argtypes = action_args + [
                ctypes.c_int, ctypes.c_char_p, ctypes.c_int
            ]
            continuous_step.restype = ctypes.c_int

    def reset(self, config: Mapping[str, object]) -> ResetFacts:
        if getattr(self, "clean_save", False):
            clean_adventure_save(getattr(self, "save_dir", None),
                                 profile_id=getattr(self, "profile_id", None),
                                 instance_id=getattr(self, "instance_id", None),
                                 template_dir=getattr(self, "template_dir", None))
        merged = dict(config)
        merged.setdefault("auto_select_seeds", self.auto_select_seeds)
        merged.setdefault("deterministic_pause", self.deterministic_pause)
        self._dll.pvz_gym_set_background_running(self.background_running)
        self._dll.pvz_gym_set_auto_skip_tutorial(self.auto_skip_tutorial)
        return super().reset(merged)

    def reset_game(self, config: Mapping[str, object] | None = None) -> Observation:
        """Compatibility wrapper; formal Gym code consumes ResetFacts via reset()."""
        facts = self.reset(dict(config or {}))
        if facts.observation is None:
            raise NativeCapabilityError(
                "reset stopped before battle; use PvZSession.wait_for_battle() after seed selection"
            )
        return facts.observation

    def wait_for_battle(self, *, timeout_ms: int = 10000,
                        deterministic_pause: bool | None = None) -> ResetFacts:
        self._dll.pvz_gym_set_background_running(self.background_running)
        self._dll.pvz_gym_set_auto_skip_tutorial(self.auto_skip_tutorial)
        return super().wait_for_battle(
            timeout_ms=timeout_ms,
            deterministic_pause=(
                self.deterministic_pause
                if deterministic_pause is None
                else bool(deterministic_pause)
            ),
        )

    def dismiss_battle_dialogs(self, *, keep_paused: bool = False) -> None:
        """Close evaluation-blocking menus without changing the stepping mode."""
        closed = False
        for dialog_id in (2, 19):  # DIALOG_NEWOPTIONS, DIALOG_PAUSED
            if self._dll.pvz_gym_get_dialog(dialog_id):
                if not self._dll.pvz_gym_kill_dialog(dialog_id):
                    raise NativeCapabilityError(f"could not close battle dialog {dialog_id}")
                closed = True
        if closed:
            # A clock resume alone leaves the modal widget on top of the Board.
            # KillDialog may unpause it, so restore the decision barrier when needed.
            clock_control = self._dll.pvz_gym_pause if keep_paused else self._dll.pvz_gym_resume
            if not clock_control():
                raise NativeCapabilityError("could not restore battle clock after closing dialogs")

    def get_game_status(self) -> GameStatus:
        data = self.get_runtime_state()
        try:
            return GameStatus(int(data.get("game_status", GameStatus.UNKNOWN)))
        except (TypeError, ValueError):
            return GameStatus.UNKNOWN

    def execute_action(self, action: Action) -> ActionResult:
        message = ctypes.create_string_buffer(512)
        code = int(self._dll.pvz_gym_execute_action(*_action_ints(action), message, len(message)))
        return ActionResult(
            success=code == 0,
            error=None if code == 0 else code,
            message=message.value.decode("utf-8", errors="replace"),
        )

    def advance_ticks(self, ticks: int) -> int:
        if ticks <= 0:
            raise ValueError(f"ticks must be positive, got {ticks}")
        advanced = int(self._dll.pvz_gym_advance_ticks(int(ticks)))
        if advanced < 0:
            raise NativeCapabilityError(f"pvz_gym_advance_ticks failed: {advanced}")
        return advanced

    def get_headless_diagnostics(self) -> dict[str, object]:
        native = getattr(self._dll, "pvz_gym_get_headless_diagnostics_json", None)
        if native is None:
            return {}
        ret, payload = _call_with_resizable_buffer(
            lambda buf, size: native(buf, size),
            initial_size=1024,
        )
        if ret < 0 or not payload:
            return {}
        try:
            data = json.loads(payload.decode("utf-8", errors="replace"))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def begin_headless_mode(self) -> bool:
        ok = bool(self._dll.pvz_gym_begin_headless_mode())
        if ok:
            return True
        diag = self.get_headless_diagnostics()
        if diag:
            raise NativeCapabilityError(
                "native headless begin failed: "
                + json.dumps(diag, ensure_ascii=False, sort_keys=True)
            )
        return False

    def advance_ticks_headless(self, ticks: int) -> int:
        if ticks <= 0:
            raise ValueError(f"ticks must be positive, got {ticks}")
        advanced = int(self._dll.pvz_gym_advance_ticks_headless(int(ticks)))
        if advanced < 0:
            raise NativeCapabilityError(f"pvz_gym_advance_ticks_headless failed: {advanced}")
        return advanced

    def step_headless(self, action: Action, ticks: int) -> StepFacts:
        if ticks <= 0:
            raise ValueError(f"ticks must be positive, got {ticks}")
        args = _action_ints(action)
        ret, payload = self._call_transition(
            lambda buf, size: self._dll.pvz_gym_env_step_headless(
                *args, int(ticks), buf, size
            ),
            initial_size=131072,
        )
        if ret < 0 or not payload:
            raise NativeCapabilityError(f"pvz_gym_env_step_headless failed: {ret}")
        return self._parse_profiled_step(payload)

    def _parse_profiled_step(self, payload):
        if bridge_timing.enabled:
            getter = getattr(self._dll, "pvz_gym_get_step_timing_json", None)
            if getter is not None:
                getter.argtypes = [ctypes.c_char_p, ctypes.c_int]
                getter.restype = ctypes.c_int
                ret_t, timing_payload = _call_with_resizable_buffer(getter, initial_size=2048)
                if ret_t >= 0:
                    for name, value in json.loads(timing_payload).items():
                        bridge_timing.add("native." + name, float(value))
        return bridge_timing.measure("bridge.parse", _parse_step_json, payload)

    def step_continuous(self, action: Action, target_ticks: int) -> StepFacts:
        if target_ticks <= 0:
            raise ValueError(f"target_ticks must be positive, got {target_ticks}")
        native_step = getattr(self._dll, "pvz_gym_env_step_continuous", None)
        if native_step is None:
            raise NativeCapabilityError(
                "continuous simulation requires a rebuilt pvz_gym_native.dll"
            )
        args = _action_ints(action)
        ret, payload = self._call_transition(
            lambda buf, size: native_step(*args, int(target_ticks), buf, size),
            initial_size=131072,
        )
        if ret < 0 or not payload:
            raise NativeCapabilityError(f"pvz_gym_env_step_continuous failed: {ret}")
        return _parse_step_json(payload)

    def end_headless_mode(self) -> bool:
        return bool(self._dll.pvz_gym_end_headless_mode())

    def _turbo_failure_details(self, operation: str) -> dict:
        details = {}
        for key, name in (("runtime_error", "pvz_gym_get_runtime_error_json"),
                          ("headless", "pvz_gym_get_headless_diagnostics_json")):
            getter = getattr(self._dll, name, None)
            if getter is None:
                continue
            try:
                ret, payload = _call_with_resizable_buffer(getter, initial_size=4096)
                value = json.loads(payload) if ret >= 0 and payload else None
                if not isinstance(value, dict) or not value:
                    continue
                if key == "runtime_error" and value.get("operation") != operation:
                    continue
                details[key] = value
            except (ValueError, TypeError, OSError, NativeCapabilityError):
                # A diagnostic failure must not hide the original transition error.
                continue
        return details

    def begin_turbo_mode(self, config: TurboConfig | None = None) -> bool:
        config = coerce_config(TurboConfig, config)
        self.get_runtime_capabilities().require_mode("turbo")
        native = getattr(self._dll, "pvz_gym_begin_turbo_mode", None)
        if native is None:
            raise NativeCapabilityError("Turbo lifecycle API is unavailable")
        try:
            self._check_runtime_status(int(native(config.native_flags)), "begin_turbo_mode")
        except NativeCapabilityError as exc:
            details = self._turbo_failure_details("begin_turbo_mode")
            if details:
                raise NativeCapabilityError(f"{exc}; native diagnostics: {details}") from exc
            raise
        self._turbo_active = True
        return True

    def step_turbo(self, action: Action, ticks: int) -> StepFacts:
        if ticks <= 0:
            raise ValueError("ticks must be positive")
        self.get_runtime_capabilities().require_mode("turbo")
        if not getattr(self, "_turbo_active", False):
            raise NativeCapabilityError("Turbo mode has not been entered")
        native = getattr(self._dll, "pvz_gym_env_step_turbo", None)
        if native is None:
            raise NativeCapabilityError("Turbo step API is unavailable")
        ret, payload = self._call_transition(
            lambda buf, size: native(*_action_ints(action), int(ticks), buf, size),
            initial_size=131072,
        )
        if ret < 0 or not payload:
            message = f"pvz_gym_env_step_turbo failed: {ret}"
            details = self._turbo_failure_details("step_turbo")
            if details:
                message += f"; native diagnostics: {details}"
            raise NativeCapabilityError(message)
        return self._parse_profiled_step(payload)

    def advance_phase(self, request: PhaseAdvanceRequest) -> PhaseAdvanceResult:
        native = getattr(self._dll, "pvz_gym_advance_phase", None)
        if native is None: raise NativeCapabilityError("phase advance API is unavailable")
        native.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        native.restype = ctypes.c_int
        ret, payload = self._call_transition(
            lambda buf, size: native(int(request.expected_phase), request.updates, buf, size), initial_size=512)
        if ret < 0: raise NativeCapabilityError("phase advance failed")
        data = decode_payload(payload)
        self._check_runtime_status(int(data["status"]), "advance_phase")
        ticks = int(data["game_ticks_advanced"])
        return PhaseAdvanceResult(RuntimePhase(data["phase_before"]), RuntimePhase(data["phase_after"]),
                                  int(data["updates_executed"]), ticks if ticks >= 0 else None)

    def read_snapshot(self) -> RuntimeSnapshot:
        native = getattr(self._dll, "pvz_gym_read_runtime_snapshot", None)
        if native is None: raise NativeCapabilityError("runtime snapshot API is unavailable")
        native.argtypes = [ctypes.c_char_p, ctypes.c_int]; native.restype = ctypes.c_int
        ret, payload = _call_with_resizable_buffer(native, initial_size=131072)
        if ret < 0: raise NativeCapabilityError("runtime snapshot failed")
        data = decode_payload(payload)
        return RuntimeSnapshot(SnapshotIdentity(*(int(data[k]) for k in
            ("process_id", "episode_id", "board_id", "revision"))),
            _parse_observation_data(data["observation"]), tuple(bool(x) for x in data["action_mask"]))

    def end_turbo_mode(self) -> bool:
        if not getattr(self, "_turbo_active", False):
            return True
        native = getattr(self._dll, "pvz_gym_end_turbo_mode", None)
        if native is None:
            raise NativeCapabilityError("Turbo lifecycle API is unavailable")
        try:
            self._check_runtime_status(int(native()), "end_turbo_mode")
        finally:
            self._turbo_active = False
        return True

    def get_action_mask(self) -> tuple[bool, ...]:
        native_capacity = ACTION_SPACE_SIZE
        buf = (ctypes.c_uint8 * native_capacity)()
        count = ctypes.c_int(0)
        code = int(self._dll.pvz_gym_get_action_mask(buf, native_capacity, ctypes.byref(count)))
        if code != 0:
            raise NativeCapabilityError(f"pvz_gym_get_action_mask failed: {code}")
        if count.value < ACTION_SPACE_SIZE:
            raise NativeCapabilityError(
                f"native action mask too short: {count.value} < {ACTION_SPACE_SIZE}"
            )
        return tuple(bool(buf[index]) for index in range(ACTION_SPACE_SIZE))

    def close(self) -> None:
        try:
            self.end_turbo_mode()
        finally:
            try:
                native = getattr(self._dll, "pvz_gym_close_runtime", None)
                if native is not None:
                    native.argtypes, native.restype = [], ctypes.c_int
                    self._check_runtime_status(int(native()), "close_runtime")
            finally:
                self._close_legacy_state()

    def _close_legacy_state(self) -> None:
        try:
            self.end_headless_mode()
        except Exception:
            pass
        self._release_pause()
        old = getattr(self, "_old_automation", None)
        if isinstance(old, dict):
            self.set_automation_config(
                bool(old.get("enabled", True)),
                bool(old.get("auto_dialogue", True)),
                bool(old.get("auto_collect", True)),
            )
            self.set_auto_whack(bool(old.get("auto_whack", True)))
        if hasattr(self, "_old_auto_skip_tutorial"):
            self._dll.pvz_gym_set_auto_skip_tutorial(bool(self._old_auto_skip_tutorial))
        try:
            setter = getattr(self._dll, "pvz_gym_set_frame_duration", None)
            if setter is not None:
                setter(10)
        except Exception:
            pass

    def _release_pause(self) -> None:
        try:
            if self._dll.pvz_gym_is_paused():
                self._dll.pvz_gym_resume()
        except Exception:
            # close() is best-effort cleanup; a vanished game process must not
            # hide the caller's original exception.
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
