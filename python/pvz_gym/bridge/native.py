"""ctypes transport for the authoritative native PvZ Gym contract."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .base import GameFactsBackend
from .wire import decode_payload
from ..profiling import bridge_timing
from ..runtime import RuntimeCapabilities, RuntimeCapabilityError, RuntimeConfig, coerce_config
from ..types import (
    Action,
    ActionResult,
    GameStatus,
    Observation,
    ResetFacts,
    RngSeedSupport,
    StepEvents,
    StepFacts,
)


class NativeCapabilityError(RuntimeCapabilityError):
    pass


def _action_ints(action: Action) -> tuple[int, int, int, int, int, int, int]:
    type_map = {"noop": 0, "plant": 1, "shovel": 2, "cob_fire": 3,
                "interact_cell": 4}
    shovel_target_map = {"normal": 0, "pumpkin": 1}
    return (
        type_map.get(action.type, -1),
        action.card_slot,
        action.row,
        action.col,
        shovel_target_map.get(action.shovel_target, -1),
        action.target_row,
        action.target_col,
    )


def _call_with_resizable_buffer(
    call: Callable[[ctypes.Array, int], int], *, initial_size: int
) -> tuple[int, bytes]:
    size = initial_size
    for _ in range(3):
        buf = ctypes.create_string_buffer(size)
        ret = int(call(buf, size))
        if ret >= 0:
            return ret, buf.raw[:ret]
        if ret < -1:
            required = -ret
            if required <= size:
                raise NativeCapabilityError(
                    f"native buffer contract returned invalid required size: {required} <= {size}"
                )
            size = required
            continue
        return ret, b""
    raise NativeCapabilityError("native JSON buffer resize retry exhausted")


def _call_transition_with_resizable_buffer(
    execute: Callable[[ctypes.Array, int], int],
    read_result: Callable[[ctypes.Array, int], int], *, initial_size: int,
) -> tuple[int, bytes]:
    # The mutating call is executed exactly once. Only the result getter may retry.
    buf = ctypes.create_string_buffer(initial_size)
    ret = int(execute(buf, initial_size))
    if ret >= 0:
        return ret, buf.raw[:ret]
    if ret == -1:
        return ret, b""
    if -ret <= initial_size:
        raise NativeCapabilityError("native transition returned an invalid required size")
    return _call_with_resizable_buffer(read_result, initial_size=-ret)


class NativePvzBackend(GameFactsBackend):
    def __init__(self, dll_path: str | Path | None = None):
        root = Path(__file__).resolve().parents[3]
        path = Path(dll_path) if dll_path else root / "bin" / "pvz_gym_native.dll"
        if not path.is_file():
            raise NativeCapabilityError(f"native PvZ Gym DLL not found: {path}")
        self._dll = ctypes.WinDLL(str(path))
        self._declare_signatures()

    def _declare_signatures(self) -> None:
        d = self._dll
        for name, args, result in (
            ("pvz_gym_begin_result_request", [], ctypes.c_uint64),
            ("pvz_gym_read_result_request", [ctypes.c_uint64, ctypes.c_char_p, ctypes.c_int], ctypes.c_int),
            ("pvz_gym_get_runtime_capabilities_json", [ctypes.c_char_p, ctypes.c_int], ctypes.c_int),
            ("pvz_gym_get_runtime_error_json", [ctypes.c_char_p, ctypes.c_int], ctypes.c_int),
            ("pvz_gym_configure_runtime", [ctypes.c_uint32, ctypes.c_int, ctypes.c_int, ctypes.c_int], ctypes.c_int),
            ("pvz_gym_begin_turbo_mode", [ctypes.c_uint32], ctypes.c_int),
            ("pvz_gym_end_turbo_mode", [], ctypes.c_int),
            ("pvz_gym_env_step_turbo", [ctypes.c_int] * 8 + [ctypes.c_char_p, ctypes.c_int], ctypes.c_int),
        ):
            function = getattr(d, name, None)
            if function is not None:
                function.argtypes, function.restype = args, result
        d.pvz_gym_attach.argtypes = []
        d.pvz_gym_attach.restype = ctypes.c_bool
        d.pvz_gym_is_game_on.argtypes = []
        d.pvz_gym_is_game_on.restype = ctypes.c_bool
        d.pvz_gym_get_ui.argtypes = []
        d.pvz_gym_get_ui.restype = ctypes.c_int
        d.pvz_gym_get_runtime_phase.argtypes = []
        d.pvz_gym_get_runtime_phase.restype = ctypes.c_int
        d.pvz_gym_get_runtime_state_json.argtypes = [ctypes.c_char_p, ctypes.c_int]
        d.pvz_gym_get_runtime_state_json.restype = ctypes.c_int
        d.pvz_gym_select_adventure_level.argtypes = [ctypes.c_int, ctypes.c_int]
        d.pvz_gym_select_adventure_level.restype = ctypes.c_int
        d.pvz_gym_select_adventure_level_id.argtypes = [ctypes.c_int]
        d.pvz_gym_select_adventure_level_id.restype = ctypes.c_int
        d.pvz_gym_get_current_adventure_level.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ]
        d.pvz_gym_get_current_adventure_level.restype = ctypes.c_int
        d.pvz_gym_error_message.argtypes = [ctypes.c_int]
        d.pvz_gym_error_message.restype = ctypes.c_char_p

        d.pvz_gym_pause.argtypes = []
        d.pvz_gym_pause.restype = ctypes.c_bool
        d.pvz_gym_resume.argtypes = []
        d.pvz_gym_resume.restype = ctypes.c_bool
        d.pvz_gym_is_paused.argtypes = []
        d.pvz_gym_is_paused.restype = ctypes.c_bool
        d.pvz_gym_step_one_tick.argtypes = []
        d.pvz_gym_step_one_tick.restype = ctypes.c_bool
        d.pvz_gym_advance_ticks.argtypes = [ctypes.c_int]
        d.pvz_gym_advance_ticks.restype = ctypes.c_int
        d.pvz_gym_get_game_clock.argtypes = []
        d.pvz_gym_get_game_clock.restype = ctypes.c_int64

        action_args = [ctypes.c_int] * 7
        d.pvz_gym_can_execute_action.argtypes = action_args
        d.pvz_gym_can_execute_action.restype = ctypes.c_int
        d.pvz_gym_execute_action.argtypes = action_args + [ctypes.c_char_p, ctypes.c_int]
        d.pvz_gym_execute_action.restype = ctypes.c_int
        d.pvz_gym_get_action_mask.argtypes = [
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.POINTER(ctypes.c_int)
        ]
        d.pvz_gym_get_action_mask.restype = ctypes.c_int
        d.pvz_gym_get_observation_json.argtypes = [ctypes.c_char_p, ctypes.c_int]
        d.pvz_gym_get_observation_json.restype = ctypes.c_int

        d.pvz_gym_env_reset_v2.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_bool,
            ctypes.c_bool, ctypes.c_bool, ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
            ctypes.c_bool, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
        ]
        d.pvz_gym_env_reset_v2.restype = ctypes.c_int
        d.pvz_gym_env_wait_for_battle.argtypes = [
            ctypes.c_bool, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
        ]
        d.pvz_gym_env_wait_for_battle.restype = ctypes.c_int
        d.pvz_gym_env_step.argtypes = action_args + [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        d.pvz_gym_env_step.restype = ctypes.c_int
        d.pvz_gym_get_rng_seed_support.argtypes = []
        d.pvz_gym_get_rng_seed_support.restype = ctypes.c_int

        d.pvz_gym_set_automation_config.argtypes = [ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
        d.pvz_gym_set_automation_config.restype = None
        d.pvz_gym_get_automation_config.argtypes = [
            ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_bool)
        ]
        d.pvz_gym_get_automation_config.restype = None
        d.pvz_gym_set_auto_whack.argtypes = [ctypes.c_bool]
        d.pvz_gym_set_auto_whack.restype = None
        d.pvz_gym_get_auto_whack.argtypes = []
        d.pvz_gym_get_auto_whack.restype = ctypes.c_bool
        d.pvz_gym_handle_prebattle.argtypes = []
        d.pvz_gym_handle_prebattle.restype = ctypes.c_bool
        d.pvz_gym_automation_collect.argtypes = []
        d.pvz_gym_automation_collect.restype = ctypes.c_int
        d.pvz_gym_automation_whack.argtypes = []
        d.pvz_gym_automation_whack.restype = ctypes.c_bool
        d.pvz_gym_automation_tick.argtypes = []
        d.pvz_gym_automation_tick.restype = ctypes.c_int

        if hasattr(d, "pvz_gym_set_process_audio_muted"):
            d.pvz_gym_set_process_audio_muted.argtypes = [ctypes.c_uint32, ctypes.c_bool]
            d.pvz_gym_set_process_audio_muted.restype = ctypes.c_int

    def set_process_audio_muted(self, process_id: int, mute: bool = True) -> bool:
        fn = getattr(self._dll, "pvz_gym_set_process_audio_muted", None)
        if fn is None:
            return False
        return fn(ctypes.c_uint32(process_id), ctypes.c_bool(mute)) == 0


    def _call_transition(self, call, *, initial_size):
        begin = getattr(self._dll, "pvz_gym_begin_result_request", None)
        read = getattr(self._dll, "pvz_gym_read_result_request", None)
        if begin is None or read is None:
            raise NativeCapabilityError(
                "safe step/reset requires a rebuilt pvz_gym_native.dll (result-request API)"
            )
        request_id = int(begin())
        if not request_id:
            raise NativeCapabilityError("native result request could not be started")
        return bridge_timing.measure("bridge.transition", _call_transition_with_resizable_buffer,
            call, lambda buf, size: read(request_id, buf, size), initial_size=initial_size,
        )

    def get_runtime_capabilities(self) -> RuntimeCapabilities:
        cached = getattr(self, "_runtime_capabilities", None)
        if cached is not None:
            return cached
        native = getattr(self._dll, "pvz_gym_get_runtime_capabilities_json", None)
        if native is None:
            # Old DLLs advertise only complete groups of existing exports. In
            # particular, interface absence never implies Turbo/PID support.
            modes, features = set(), set()
            groups = {
                "stepped": ("pvz_gym_env_step",),
                "headless": ("pvz_gym_begin_headless_mode", "pvz_gym_end_headless_mode", "pvz_gym_env_step_headless"),
                "continuous": ("pvz_gym_env_step_continuous",),
            }
            for mode, names in groups.items():
                if all(callable(getattr(self._dll, name, None)) for name in names):
                    modes.add(mode)
            if all(callable(getattr(self._dll, name, None)) for name in
                   ("pvz_gym_begin_result_request", "pvz_gym_read_result_request")):
                features.add("result_requests")
            return RuntimeCapabilities(modes=frozenset(modes), features=frozenset(features))
        ret, payload = _call_with_resizable_buffer(native, initial_size=2048)
        if ret < 0 or not payload:
            raise NativeCapabilityError("native runtime capability query failed")
        try:
            data = json.loads(payload)
        except (ValueError, UnicodeError) as exc:
            raise NativeCapabilityError("invalid native runtime capabilities JSON") from exc
        if not isinstance(data, dict):
            raise NativeCapabilityError("native runtime capabilities must be an object")
        self._runtime_capabilities = RuntimeCapabilities.from_mapping(data)
        return self._runtime_capabilities

    @staticmethod
    def _check_runtime_status(code: int, operation: str) -> None:
        if code != 0:
            status = {1: "not implemented", 2: "invalid argument", 3: "invalid state"}.get(code, f"error {code}")
            raise NativeCapabilityError(f"{operation}: {status}")

    def configure_runtime(self, config: RuntimeConfig) -> None:
        config = coerce_config(RuntimeConfig, config)
        self.get_runtime_capabilities().require_config(config)
        native = getattr(self._dll, "pvz_gym_configure_runtime", None)
        if native is None:
            if config != RuntimeConfig():
                raise NativeCapabilityError("runtime configuration API is unavailable")
            return
        code = int(native(
            config.process_id or 0,
            {"legacy": 0, "resident": 1}[config.executor],
            {"json": 0, "binary": 1, "shared_memory": 2}[config.transport],
            {"legacy": 0, "shared": 1}[config.snapshot],
        ))
        self._check_runtime_status(code, "configure_runtime")
        self.runtime_config = config

    def get_raw_scene(self) -> int:
        return int(self._dll.pvz_gym_get_ui())

    def attach(self) -> bool:
        return bool(self._dll.pvz_gym_attach())

    def is_game_on(self) -> bool:
        return bool(self._dll.pvz_gym_is_game_on())

    def get_runtime_phase(self) -> int:
        return int(self._dll.pvz_gym_get_runtime_phase())

    def get_runtime_state(self) -> dict[str, object]:
        ret, payload = _call_with_resizable_buffer(
            lambda buf, size: self._dll.pvz_gym_get_runtime_state_json(buf, size),
            initial_size=4096,
        )
        if ret < 0 or not payload:
            raise NativeCapabilityError(f"pvz_gym_get_runtime_state_json failed: {ret}")
        return json.loads(payload.decode("utf-8", errors="replace"))

    def set_automation_config(self, enabled: bool = True, auto_dialogue: bool = True,
                              auto_collect: bool = True) -> None:
        self._dll.pvz_gym_set_automation_config(bool(enabled), bool(auto_dialogue), bool(auto_collect))

    def get_automation_config(self) -> dict[str, bool]:
        e, d, c = ctypes.c_bool(), ctypes.c_bool(), ctypes.c_bool()
        self._dll.pvz_gym_get_automation_config(ctypes.byref(e), ctypes.byref(d), ctypes.byref(c))
        return {"enabled": bool(e.value), "auto_dialogue": bool(d.value),
                "auto_collect": bool(c.value), "auto_whack": self.get_auto_whack()}

    def set_auto_whack(self, enabled: bool) -> None:
        self._dll.pvz_gym_set_auto_whack(bool(enabled))

    def get_auto_whack(self) -> bool:
        return bool(self._dll.pvz_gym_get_auto_whack())

    def handle_prebattle(self) -> bool:
        return bool(self._dll.pvz_gym_handle_prebattle())

    def automation_collect(self) -> int:
        return int(self._dll.pvz_gym_automation_collect())

    def automation_whack(self) -> bool:
        return bool(self._dll.pvz_gym_automation_whack())

    def automation_tick(self) -> int:
        return int(self._dll.pvz_gym_automation_tick())

    def get_rng_seed_support(self) -> RngSeedSupport:
        try:
            return RngSeedSupport(int(self._dll.pvz_gym_get_rng_seed_support()))
        except ValueError:
            return RngSeedSupport.UNSUPPORTED

    def reset(self, config: Mapping[str, object]) -> ResetFacts:
        level_id = int(config.get("level_id", 1))
        playthrough = int(config.get("playthrough", 0) or 0)
        seed = int(config.get("seed", 0))
        has_seed = "seed" in config and config.get("seed") is not None
        stop_at_seed_selection = bool(config.get("stop_at_seed_selection", False))
        auto_select_seeds = bool(config.get("auto_select_seeds", True))
        deterministic_pause = bool(config.get("deterministic_pause", True))
        timeout = int(config.get("timeout_ms", 10000))

        raw_seed_types = config.get("seed_types")
        seed_types: tuple[int, ...] = ()
        if isinstance(raw_seed_types, Sequence) and not isinstance(raw_seed_types, (str, bytes)):
            seed_types = tuple(int(value) for value in raw_seed_types)
        seed_array = (ctypes.c_int32 * len(seed_types))(*seed_types) if seed_types else None

        def invoke(buf, size):
            return self._dll.pvz_gym_env_reset_v2(
                level_id,
                playthrough,
                seed,
                has_seed,
                stop_at_seed_selection,
                auto_select_seeds,
                seed_array,
                len(seed_types),
                deterministic_pause,
                timeout,
                buf,
                size,
            )

        ret, payload = self._call_transition(invoke, initial_size=65536)
        if ret < 0 or not payload:
            raise NativeCapabilityError(f"pvz_gym_env_reset_v2 failed: {ret}")
        return _parse_reset_result(payload)

    def wait_for_battle(self, *, timeout_ms: int = 10000,
                        deterministic_pause: bool = True) -> ResetFacts:
        ret, payload = self._call_transition(
            lambda buf, size: self._dll.pvz_gym_env_wait_for_battle(
                bool(deterministic_pause), int(timeout_ms), buf, size
            ),
            initial_size=65536,
        )
        if ret < 0 or not payload:
            raise NativeCapabilityError(f"pvz_gym_env_wait_for_battle failed: {ret}")
        return _parse_reset_result(payload)

    def step(self, action: Action, ticks: int) -> StepFacts:
        args = _action_ints(action)
        ret, payload = self._call_transition(
            lambda buf, size: self._dll.pvz_gym_env_step(*args, int(ticks), buf, size),
            initial_size=131072,
        )
        if ret < 0 or not payload:
            raise NativeCapabilityError(f"pvz_gym_env_step failed: {ret}")
        return _parse_step_json(payload)

    def get_observation(self) -> Observation:
        if not self._dll.pvz_gym_attach():
            raise NativeCapabilityError("PlantsVsZombies.exe is not attached")
        ret, payload = _call_with_resizable_buffer(
            lambda buf, size: self._dll.pvz_gym_get_observation_json(buf, size),
            initial_size=65536,
        )
        if ret < 0 or not payload:
            raise NativeCapabilityError(f"no native observation available: {ret}")
        return _parse_observation_json(payload.decode("utf-8", errors="replace"))


def _parse_observation_data(data: Mapping[str, object]) -> Observation:
    return Observation(
        global_state=data.get("global", {}), cards=data.get("cards", []),
        plants=data.get("plants", []), zombies=data.get("zombies", []),
        dropped_cards=data.get("dropped_cards", []), grid_items=data.get("grid_items", []),
        ice_trails=data.get("ice_trails", []), mowers=data.get("mowers", []), grid=data.get("grid", []),
        zombie_health=data.get("zombie_health", []),
        boss_projectiles=data.get("boss_projectiles", []),
        schema_version=int(data.get("schema_version", 3)),
        quality=str(data.get("quality", "fresh")),
    )


def _parse_observation_json(json_str: str) -> Observation:
    return _parse_observation_data(json.loads(json_str))


def _parse_reset_result(json_str) -> ResetFacts:
    data = decode_payload(json_str)
    if not bool(data.get("success", False)):
        raise NativeCapabilityError(
            f"native reset failed: {data.get('error')} {data.get('message', '')}".strip()
        )
    observation_data = data.get("observation")
    observation = _parse_observation_data(observation_data) if isinstance(observation_data, Mapping) else None
    try:
        status = GameStatus(int(data.get("status", GameStatus.UNKNOWN)))
    except ValueError:
        status = GameStatus.UNKNOWN
    try:
        rng_support = RngSeedSupport(int(data.get("rng_seed_support", 0)))
    except ValueError:
        rng_support = RngSeedSupport.UNSUPPORTED
    return ResetFacts(
        observation=observation,
        game_clock=int(data.get("game_clock", 0)),
        status=status,
        rng_seed_support=rng_support,
        rng_seed_applied=bool(data.get("rng_seed_applied", False)),
    )


def _parse_step_json(json_str) -> StepFacts:
    data = decode_payload(json_str)
    if not bool(data.get("success", False)):
        raise NativeCapabilityError(
            f"native step failed: {data.get('error')} {data.get('message', '')}".strip()
        )
    ev = data.get("events", {})
    ar = data.get("action", {})
    observation_data = data.get("observation")
    if not isinstance(observation_data, Mapping) or not isinstance(observation_data.get("global"), Mapping):
        raise NativeCapabilityError("successful native step omitted its required observation/global")
    obs = _parse_observation_data(observation_data)
    try:
        status = GameStatus(int(data.get("status", 0)))
    except ValueError:
        status = GameStatus.UNKNOWN
    fallback = bool(data.get("terminal_observation_fallback", False))
    quality = str(data.get("observation_quality", "terminal_fallback" if fallback else obs.quality))
    exact = bool(data.get("ticks_advanced_exact", not fallback)) and not fallback and quality == "fresh"
    replay_eligible = bool(data.get("replay_eligible", not fallback)) and quality == "fresh" and exact
    if obs.schema_version >= 4:
        if int(obs.global_state.get("status", GameStatus.UNKNOWN)) != int(status):
            raise NativeCapabilityError("native transition and observation status disagree")
        if quality != obs.quality:
            raise NativeCapabilityError("native transition and observation quality disagree")
        if quality == "fresh" and int(obs.global_state.get("game_clock", -1)) != int(data.get("game_clock", -2)):
            raise NativeCapabilityError("fresh native transition and observation clocks disagree")
    return StepFacts(
        observation=obs,
        events=StepEvents(
            zombies_killed=int(ev.get("zombies_killed", 0)),
            plants_lost=int(ev.get("plants_lost", 0)),
            sun_collected=int(ev.get("sun_collected", 0)),
            sun_spent=int(ev.get("sun_spent", 0)),
            mowers_used=int(ev.get("mowers_used", 0)),
            mower_triggered_rows=tuple(int(row) for row in ev.get("mower_triggered_rows", ())),
            waves_advanced=int(ev.get("waves_advanced", 0)),
            won=bool(ev.get("won", False)),
            lost=bool(ev.get("lost", False)),
        ),
        action=ActionResult(
            success=bool(ar.get("success", False)),
            error=ar.get("error"),
            message=str(ar.get("message", "")),
        ),
        action_mask=tuple(bool(v) for v in data.get("action_mask", [])),
        status=status,
        game_clock=int(data.get("game_clock", 0)),
        automation_collected=int(data.get("automation_collected", 0)),
        terminal_observation_fallback=fallback,
        observation_quality=quality,
        last_confirmed_game_clock=int(data.get("last_confirmed_game_clock", data.get("game_clock", 0))),
        ticks_advanced=int(data["ticks_advanced"]) if "ticks_advanced" in data else None,
        ticks_advanced_exact=exact,
        replay_eligible=replay_eligible,
    )


def set_process_audio_muted(process_id: int, mute: bool = True, dll_path: str | Path | None = None) -> bool:
    """即时对指定进程 ID 开启或关闭 WASAPI 静音"""
    try:
        backend = NativePvzBackend(dll_path=dll_path)
        return backend.set_process_audio_muted(process_id, mute)
    except Exception:
        return False
