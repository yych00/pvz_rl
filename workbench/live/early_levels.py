# -*- coding: utf-8 -*-
"""Adventure-1 / Adventure-2 LIVE contracts for levels 1-2 and 1-3.

Playthrough identity is authoritative: it comes from native RuntimeState field
``finished_adventure`` (PlayerInfo::mFinishedAdventure), not from whether a
SeedChooser happened to appear.

Adventure 1 uses the historical early-game card priorities for each level.
Adventure 2+ uses the richer unlocked-card priorities while preserving any
Dave-forced/disabled selections exposed by the native chooser state.
"""

from __future__ import annotations

import ctypes
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence
from pvz_gym.default_decks import deck_counter_gaps, plan_default_deck
from pvz_gym.seed_types import SeedOptionState, SeedSelectionState

from .level_1_1_live_tests import (
    ACTION_PLANT,
    ACTION_SPACE_SIZE,
    PHASE_AWARD,
    PHASE_BATTLE,
    PHASE_COMPLETE,
    PHASE_DIALOGUE,
    PHASE_INTRO,
    PHASE_LOST,
    PHASE_SEED,
    SEED_CAPACITY,
    SEED_DESELECT_BASE,
    SEED_CONFIRM_ACTION,
    _emit,
    _observation,
    _runtime,
    _seed_mask,
    _seed_state,
    _terminal_outcome,
    _wait_phase,
    _wait_prebattle_ready,
)
from .level_1_2_policy import (
    LEVEL_1_2_FIRST_RUN_DECK_PRIORITY,
    LEVEL_1_2_REPLAY_DECK_PRIORITY,
    choose_level_1_2_first_run_plant,
    choose_level_1_2_replay_plant,
)
from .level_1_3_policy import (
    LEVEL_1_3_FIRST_RUN_DECK_PRIORITY,
    LEVEL_1_3_REPLAY_DECK_PRIORITY,
    choose_level_1_3_first_run_plant,
    choose_level_1_3_replay_plant,
)

ProgressFn = Optional[Callable[[str, str], None]]
PolicyFn = Callable[[List[bool], dict], Optional[tuple]]

PLAYTHROUGH_AUTO = "AUTO"
PLAYTHROUGH_1 = "ADVENTURE_1"
PLAYTHROUGH_2 = "ADVENTURE_2"

SIMULATION_STEPPED = "stepped"
SIMULATION_ANIMATED = "animated"
SIMULATION_HEADLESS = "headless"
SIMULATION_CONTINUOUS = "continuous"

ACTION_INTERACT_CELL = 4
INTERACT_BASE = 703

# Backward-compatible names from the earlier first-run/replay draft.
PATH_AUTO = PLAYTHROUGH_AUTO
PATH_FIRST_RUN = PLAYTHROUGH_1
PATH_REPLAY = PLAYTHROUGH_2


@dataclass(frozen=True)
class EarlyLevelSpec:
    area: int
    sub_level: int
    level_id: int
    label: str
    adventure1_priority: Sequence[int]
    adventure2_priority: Sequence[int]
    adventure1_policy: PolicyFn
    adventure2_policy: PolicyFn
    max_plants: int = 40


LEVEL_1_2_SPEC = EarlyLevelSpec(
    area=1,
    sub_level=2,
    level_id=2,
    label="1-2",
    adventure1_priority=tuple(LEVEL_1_2_FIRST_RUN_DECK_PRIORITY),
    adventure2_priority=tuple(LEVEL_1_2_REPLAY_DECK_PRIORITY),
    adventure1_policy=choose_level_1_2_first_run_plant,
    adventure2_policy=choose_level_1_2_replay_plant,
    max_plants=35,
)

LEVEL_1_3_SPEC = EarlyLevelSpec(
    area=1,
    sub_level=3,
    level_id=3,
    label="1-3",
    adventure1_priority=tuple(LEVEL_1_3_FIRST_RUN_DECK_PRIORITY),
    adventure2_priority=tuple(LEVEL_1_3_REPLAY_DECK_PRIORITY),
    adventure1_policy=choose_level_1_3_first_run_plant,
    adventure2_policy=choose_level_1_3_replay_plant,
    max_plants=40,
)


@dataclass
class EarlyLevelE2EResult:
    success: bool
    level: str
    expected_path: str
    actual_path: str
    outcome: str
    message: str
    elapsed_ms: float
    finished_adventure: int = -1
    ticks_advanced: int = 0
    plants_executed: int = 0
    collected_items: int = 0
    seed_actions: int = 0
    phases: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EarlyLevelStressResult:
    success: bool
    level: str
    expected_path: str
    requested_episodes: int
    completed_episodes: int
    wins: int
    losses: int
    failures: int
    path_mismatches: int
    timeouts: int
    elapsed_ms: float
    episode_results: List[EarlyLevelE2EResult] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        terminals = self.wins + self.losses
        return self.wins / terminals if terminals else 0.0


def _normalize_playthrough(value: str) -> str:
    value = str(value or PLAYTHROUGH_AUTO).strip().upper()
    aliases = {
        "FIRST_RUN": PLAYTHROUGH_1,
        "REPLAY": PLAYTHROUGH_2,
        "1": PLAYTHROUGH_1,
        "2": PLAYTHROUGH_2,
    }
    value = aliases.get(value, value)
    if value not in {PLAYTHROUGH_AUTO, PLAYTHROUGH_1, PLAYTHROUGH_2}:
        raise ValueError(
            f"expected_playthrough must be AUTO/ADVENTURE_1/ADVENTURE_2, got {value!r}"
        )
    return value


def normalize_simulation_mode(value: str) -> str:
    value = str(value or SIMULATION_STEPPED).strip().lower()
    aliases = {
        "sync": SIMULATION_STEPPED,
        "current": SIMULATION_STEPPED,
        "native": SIMULATION_ANIMATED,
        "persistent": SIMULATION_HEADLESS,
        "continuous": SIMULATION_CONTINUOUS,
        "realtime": SIMULATION_CONTINUOUS,
        "real-time": SIMULATION_CONTINUOUS,
    }
    value = aliases.get(value, value)
    if value not in {
        SIMULATION_STEPPED,
        SIMULATION_ANIMATED,
        SIMULATION_HEADLESS,
        SIMULATION_CONTINUOUS,
    }:
        raise ValueError(
            "simulation_mode must be stepped/animated/headless/continuous, "
            f"got {value!r}"
        )
    return value


def _bind_speed_native(client: Any) -> bool:
    dll = getattr(client, "dll", None)
    required = (
        "pvz_gym_get_frame_duration",
        "pvz_gym_set_frame_duration",
        "pvz_gym_begin_headless_mode",
        "pvz_gym_advance_ticks_headless",
        "pvz_gym_end_headless_mode",
    )
    if dll is None or any(not hasattr(dll, name) for name in required):
        return False
    dll.pvz_gym_get_frame_duration.argtypes = []
    dll.pvz_gym_get_frame_duration.restype = ctypes.c_int
    dll.pvz_gym_set_frame_duration.argtypes = [ctypes.c_int]
    dll.pvz_gym_set_frame_duration.restype = ctypes.c_bool
    dll.pvz_gym_begin_headless_mode.argtypes = []
    dll.pvz_gym_begin_headless_mode.restype = ctypes.c_bool
    dll.pvz_gym_advance_ticks_headless.argtypes = [ctypes.c_int]
    dll.pvz_gym_advance_ticks_headless.restype = ctypes.c_int
    dll.pvz_gym_end_headless_mode.argtypes = []
    dll.pvz_gym_end_headless_mode.restype = ctypes.c_bool
    return True


def _playthrough_from_runtime(state: dict) -> tuple[str, int]:
    try:
        finished = int(state.get("finished_adventure", -1))
    except Exception:
        finished = -1
    if finished < 0:
        return "UNKNOWN", finished
    if finished == 0:
        return PLAYTHROUGH_1, finished
    return PLAYTHROUGH_2, finished


def _selectable_seed(mask: Sequence[bool], state: dict, priority: Sequence[int]) -> Optional[int]:
    selected = {int(v) for v in state.get("selected_seeds", [])}
    seen = set()
    for plant_type in list(priority) + list(range(SEED_CAPACITY)):
        plant_type = int(plant_type)
        if plant_type in seen or plant_type in selected:
            continue
        seen.add(plant_type)
        if 0 <= plant_type < SEED_CAPACITY and bool(mask[plant_type]):
            return plant_type
    return None


def _run_seed_selection(
    client: Any,
    spec: EarlyLevelSpec,
    playthrough: str,
    progress: ProgressFn,
    phases: List[Dict[str, Any]],
    seed_priority=None,
    seed_selection_mode="adaptive",
) -> int:
    """Build a playthrough-specific deck through the original SeedChooser."""
    priority = (
        spec.adventure1_priority
        if playthrough == PLAYTHROUGH_1
        else spec.adventure2_priority
    )
    actions = 0
    state = _seed_state(client)
    chooser = SeedSelectionState(
        present=bool(state.get("present")),
        required_count=int(state.get("required_count", 0)),
        selected_seeds=tuple(state.get("selected_seeds", [])),
        options=tuple(SeedOptionState(**option) for option in state.get("options", [])),
        zombie_types_in_level=(tuple(state.get("zombie_types_in_level", []))
                               if state.get("zombie_roster_available") else None),
    )
    # Use the live preview for both playthroughs, preserving the old specialized
    # priorities when running against a DLL that lacks the new JSON fields.
    if chooser.zombie_types_in_level is not None or seed_priority is not None or seed_selection_mode in ("fixed", "fixed_shuffled", "adaptive_shuffled"):
        priority = plan_default_deck(chooser, spec.level_id,
                                     1 if playthrough == PLAYTHROUGH_1 else 2,
                                     priority_override=seed_priority,
                                     selection_mode=seed_selection_mode)
        for gap in deck_counter_gaps(chooser, spec.level_id, priority):
            _emit(progress, phases, "SEED", f"选卡应对缺口（解锁/槽位限制）：{gap}")
    else:
        _emit(progress, phases, "SEED", "僵尸预览不可用：沿用默认优先级，请更新原生 DLL")
    preferred = set(int(v) for v in priority)
    disabled_types = {
        int(opt.get("plant_type", -1))
        for opt in state.get("options", [])
        if bool(opt.get("disabled", False))
    }

    _emit(
        progress,
        phases,
        "SEED",
        (
            f"{spec.label} {playthrough} chooser: "
            f"selected={state.get('selected_count')}/{state.get('required_count')}"
        ),
    )

    # Dave-forced/disabled seeds are never removed. This matters especially on
    # Adventure 2 where the original game can force part of the deck.
    for plant_type in list(state.get("selected_seeds", [])):
        plant_type = int(plant_type)
        if plant_type in disabled_types or (plant_type in preferred and
                                              seed_selection_mode not in ("fixed", "fixed_shuffled", "adaptive_shuffled")):
            continue
        mask = _seed_mask(client)
        idx = SEED_DESELECT_BASE + plant_type
        if 0 <= idx < len(mask) and bool(mask[idx]):
            code, msg = client.deselect_seed(plant_type)
            if code != 0:
                raise RuntimeError(f"{spec.label} Deselect({plant_type}) failed: {code} {msg}")
            actions += 1

    for _ in range(SEED_CAPACITY + 4):
        state = _seed_state(client)
        if bool(state.get("can_confirm", False)):
            break
        mask = _seed_mask(client)
        plant_type = _selectable_seed(mask, state, priority)
        if plant_type is None:
            raise RuntimeError(
                f"{spec.label} {playthrough} deck not confirmable and no legal Select remains: "
                f"{state.get('selected_count')}/{state.get('required_count')}"
            )
        code, msg = client.select_seed(plant_type)
        if code != 0:
            raise RuntimeError(f"{spec.label} Select({plant_type}) failed: {code} {msg}")
        actions += 1
    else:
        raise RuntimeError(f"{spec.label} seed fill exceeded safety bound")

    ready = _seed_state(client)
    mask = _seed_mask(client)
    if not bool(ready.get("can_confirm", False)) or not bool(mask[SEED_CONFIRM_ACTION]):
        raise RuntimeError(f"{spec.label} deck and confirm mask are not both ready")
    if bool(ready.get("seeds_in_flight", False)):
        raise RuntimeError(f"{spec.label} deck still has seeds in flight")

    selected = [int(v) for v in ready.get("selected_seeds", [])]
    _emit(
        progress,
        phases,
        "SEED",
        f"{spec.label} {playthrough} deck: " + ",".join(map(str, selected)),
    )

    code, msg = client.confirm_seeds()
    if code != 0:
        raise RuntimeError(f"{spec.label} confirm_seeds failed: {code} {msg}")
    actions += 1

    state = _wait_phase(
        client,
        [PHASE_BATTLE, PHASE_COMPLETE, PHASE_AWARD, PHASE_LOST],
        timeout_s=15.0,
        level_id=spec.level_id,
    )
    if str(state.get("phase_name")) != PHASE_BATTLE:
        raise RuntimeError(
            f"{spec.label} confirm did not reach BATTLE: "
            f"{state.get('phase_name')} / {state.get('raw_scene_name')}"
        )
    _emit(progress, phases, "SEED", f"{spec.label} confirm -> BATTLE")
    return actions


def run_early_level_e2e(
    client: Any,
    spec: EarlyLevelSpec,
    *,
    expected_path: str = PLAYTHROUGH_AUTO,
    expected_playthrough: Optional[str] = None,
    progress: ProgressFn = None,
    max_ticks: int = 60000,
    step_ticks: int = 50,
    max_plants: Optional[int] = None,
    simulation_mode: str = SIMULATION_STEPPED,
    animated_frame_ms: int = 1,
    pause_event: Optional[threading.Event] = None,
    stop_event: Optional[threading.Event] = None,
    seed_priority=None,
    seed_selection_mode="adaptive",
) -> EarlyLevelE2EResult:
    """Run one playthrough-aware early-level LIVE contract."""
    if expected_playthrough is not None:
        expected_path = expected_playthrough
    expected_path = _normalize_playthrough(expected_path)
    max_plants = spec.max_plants if max_plants is None else int(max_plants)

    t0 = time.perf_counter()
    phases: List[Dict[str, Any]] = []
    actual_path = "UNKNOWN"
    finished_adventure = -1
    ticks_advanced = plants_executed = collected_items = seed_actions = 0
    old_automation: Optional[dict] = None
    original_paused = False
    original_frame_ms = -1
    headless_active = False

    def finish(success: bool, outcome: str, message: str) -> EarlyLevelE2EResult:
        return EarlyLevelE2EResult(
            success=success,
            level=spec.label,
            expected_path=expected_path,
            actual_path=actual_path,
            outcome=outcome,
            message=message,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            finished_adventure=finished_adventure,
            ticks_advanced=ticks_advanced,
            plants_executed=plants_executed,
            collected_items=collected_items,
            seed_actions=seed_actions,
            phases=phases,
        )

    try:
        if step_ticks <= 0 or max_ticks <= 0 or max_plants <= 0:
            return finish(False, "CONFIG_ERROR", "step_ticks/max_ticks/max_plants must be > 0")
        simulation_mode = normalize_simulation_mode(simulation_mode)
        animated_frame_ms = int(animated_frame_ms)
        if animated_frame_ms <= 0:
            return finish(False, "CONFIG_ERROR", "animated_frame_ms must be > 0")
        if not getattr(client, "is_loaded", False):
            return finish(False, "NOT_STARTED", "DLL 未加载")

        probe = _runtime(client)
        if probe.get("error"):
            return finish(False, "API_ERROR", str(probe.get("error")))

        on, _ = client.is_game_on()
        if not on:
            ok, msg = client.attach()
            _emit(progress, phases, "ATTACH", f"attach={ok}: {msg}")
            on, _ = client.is_game_on()
        if not on:
            if stop_event is not None:
                stop_event.set()
            return finish(False, "ABORTED", "未找到运行中的 PlantsVsZombies.exe，自动终止")

        profile_state = _runtime(client)
        actual_path, finished_adventure = _playthrough_from_runtime(profile_state)
        if actual_path == "UNKNOWN":
            return finish(
                False,
                "PLAYTHROUGH_UNKNOWN",
                "native runtime_state has no valid finished_adventure; rebuild pvz_gym_native.dll",
            )
        if expected_path != PLAYTHROUGH_AUTO and actual_path != expected_path:
            return finish(
                False,
                "PATH_MISMATCH",
                (
                    f"期望 {expected_path}，但 PlayerInfo::mFinishedAdventure="
                    f"{finished_adventure} => {actual_path}"
                ),
            )

        old_automation = client.get_automation_config()
        client.set_automation_config(True, True, True)

        _emit(
            progress,
            phases,
            "LEVEL",
            (
                f"切换到冒险模式 {spec.label}; playthrough={actual_path}; "
                f"finished_adventure={finished_adventure}"
            ),
        )
        code, msg = client.select_level(spec.area, spec.sub_level)
        if code != 0:
            return finish(False, "LEVEL_ERROR", f"select_level({spec.area},{spec.sub_level}) failed: {code} {msg}")

        state = _wait_phase(
            client,
            [PHASE_INTRO, PHASE_DIALOGUE, PHASE_SEED, PHASE_BATTLE],
            timeout_s=5.0,
            level_id=spec.level_id,
        )
        if str(state.get("phase_name")) == "OFFLINE" or not state.get("process_available", True):
            if stop_event is not None:
                stop_event.set()
            return finish(False, "ABORTED", "游戏异常退出，自动触发终止")
        if state.get("error"):
            return finish(False, "PHASE_ERROR", str(state.get("error")))

        if str(state.get("phase_name")) in (PHASE_INTRO, PHASE_DIALOGUE):
            _emit(progress, phases, "PREBATTLE", f"推进 {spec.label} intro/Dave")
            client.handle_prebattle()
            state = _wait_prebattle_ready(
                client,
                safety_timeout_s=60.0,
                level_id=spec.level_id,
            )
            if str(state.get("phase_name")) == "OFFLINE" or not state.get("process_available", True):
                if stop_event is not None:
                    stop_event.set()
                return finish(False, "ABORTED", "游戏异常退出，自动触发终止")

        # Re-read the profile count after level transition as an integrity check.
        transitioned_playthrough, transitioned_finished = _playthrough_from_runtime(state)
        if transitioned_playthrough != "UNKNOWN":
            if transitioned_playthrough != actual_path:
                return finish(
                    False,
                    "PLAYTHROUGH_CHANGED",
                    (
                        f"playthrough changed during level transition: "
                        f"{actual_path}->{transitioned_playthrough}"
                    ),
                )
            finished_adventure = transitioned_finished

        phase = str(state.get("phase_name", "UNKNOWN"))
        if actual_path == PLAYTHROUGH_1:
            policy = spec.adventure1_policy
        else:
            policy = spec.adventure2_policy

        if phase == PHASE_SEED:
            _emit(progress, phases, "PATH", f"{spec.label}: {actual_path} with SeedChooser")
            seed_actions = _run_seed_selection(
                client,
                spec,
                actual_path,
                progress,
                phases,
                seed_priority=seed_priority,
                seed_selection_mode=seed_selection_mode,
            )
        elif phase == PHASE_BATTLE:
            _emit(progress, phases, "PATH", f"{spec.label}: {actual_path} direct battle")
        else:
            return finish(False, "PHASE_ERROR", f"prebattle ended in unexpected phase={phase}")

        obs, msg = _observation(client)
        if not obs:
            return finish(False, "OBS_ERROR", f"battle observation unavailable: {msg}")
        gl = obs.get("global", {})
        if int(gl.get("level_id", -1)) != spec.level_id:
            return finish(
                False,
                "LEVEL_ERROR",
                f"observation level_id={gl.get('level_id')}, expected {spec.level_id}",
            )
        is_conveyor_level = (
            bool(gl.get("is_conveyor"))
            or gl.get("card_system") == "conveyor"
            or spec.level_id in (5, 10, 20, 25, 30, 40, 45, 50)
        )
        is_dropped_level = (
            gl.get("card_system") == "dropped"
            or spec.level_id == 35
        )
        if not obs.get("cards") and not is_conveyor_level and not is_dropped_level:
            return finish(False, "OBS_ERROR", "battle cards are empty")

        card_types = [
            int(c.get("plant_type", -1))
            for c in obs.get("cards", [])
            if isinstance(c, dict)
        ]
        _emit(
            progress,
            phases,
            "BATTLE",
            f"ready {actual_path}, sun={gl.get('sun')}, cards={card_types}",
        )

        original_paused = bool(client.is_paused())
        if simulation_mode == SIMULATION_STEPPED:
            if not original_paused:
                client.pause()
            if not client.is_paused():
                return finish(False, "CLOCK_ERROR", "cannot pause battle")
            _emit(progress, phases, "CLOCK", "stepped/sync; advance_ticks drives game time")
        elif simulation_mode == SIMULATION_CONTINUOUS:
            if client.is_paused() and not client.resume():
                return finish(False, "CLOCK_ERROR", "cannot start continuous native loop")
            _emit(
                progress,
                phases,
                "CLOCK",
                "continuous/native loop; no deterministic pause and no frame override",
            )
        else:
            if not _bind_speed_native(client):
                return finish(False, "CLOCK_ERROR", "speed/headless native exports unavailable; rebuild DLL")
            original_frame_ms = int(client.dll.pvz_gym_get_frame_duration())
            if original_frame_ms <= 0:
                return finish(False, "CLOCK_ERROR", "cannot read engine frame duration")
            if simulation_mode == SIMULATION_HEADLESS:
                headless_active = bool(client.dll.pvz_gym_begin_headless_mode())
                if not headless_active:
                    return finish(False, "CLOCK_ERROR", "cannot enter persistent headless mode")
                _emit(progress, phases, "CLOCK", "headless/persistent; render loop frozen between decisions")
            else:
                if not bool(client.dll.pvz_gym_set_frame_duration(animated_frame_ms)):
                    return finish(False, "CLOCK_ERROR", f"cannot set animated frame to {animated_frame_ms} ms")
                if not client.resume():
                    return finish(False, "CLOCK_ERROR", "cannot start animated native loop")
                _emit(
                    progress,
                    phases,
                    "CLOCK",
                    f"animated/native loop; frame={animated_frame_ms} ms",
                )

        next_progress_tick = 1000
        while ticks_advanced < max_ticks:
            if stop_event is not None and stop_event.is_set():
                return finish(False, "ABORTED", "用户终止")
            if pause_event is not None:
                while not pause_event.is_set():
                    if simulation_mode in (SIMULATION_ANIMATED, SIMULATION_CONTINUOUS) and not client.is_paused():
                        client.pause()
                    if stop_event is not None and stop_event.is_set():
                        return finish(False, "ABORTED", "用户终止")
                    time.sleep(0.05)
                if simulation_mode in (SIMULATION_ANIMATED, SIMULATION_CONTINUOUS) and client.is_paused():
                    client.resume()

            state = _runtime(client)
            if str(state.get("phase_name")) == "OFFLINE" or not state.get("process_available", True):
                if stop_event is not None:
                    stop_event.set()
                return finish(False, "ABORTED", "游戏异常退出，自动触发终止")
            if state.get("error"):
                return finish(False, "API_ERROR", str(state.get("error")))
            outcome = _terminal_outcome(state)
            phase = str(state.get("phase_name", "UNKNOWN"))
            if outcome:
                _emit(progress, phases, "TERMINAL", f"{outcome}: phase={phase}")
                if plants_executed <= 0:
                    return finish(False, outcome, "terminal reached without a successful Plant action")
                return finish(
                    True,
                    outcome,
                    (
                        f"{spec.label} {actual_path} LIVE PASS: terminal={outcome}, "
                        f"seed_actions={seed_actions}, plants={plants_executed}, "
                        f"ticks={ticks_advanced}, collected={collected_items}"
                    ),
                )
            if phase != PHASE_BATTLE:
                return finish(False, "STATE_ERROR", f"unexpected battle-loop phase={phase}")

            obs, msg = _observation(client)
            if not obs:
                on, _ = client.is_game_on()
                if not on:
                    if stop_event is not None:
                        stop_event.set()
                    return finish(False, "ABORTED", "游戏异常退出，自动触发终止")
                return finish(False, "OBS_ERROR", f"battle observation unavailable: {msg}")

            collected = int(client.automation_tick())
            if collected < 0:
                return finish(False, "AUTOMATION_ERROR", f"automation_tick returned {collected}")
            collected_items += collected

            if plants_executed < max_plants:
                code, mask, count, mask_msg = client.get_action_mask(ACTION_SPACE_SIZE)
                if code != 0 or count != ACTION_SPACE_SIZE or len(mask) != ACTION_SPACE_SIZE:
                    return finish(
                        False,
                        "MASK_ERROR",
                        f"action mask invalid: code={code}, count={count}, {mask_msg}",
                    )

                choice = policy(list(mask), obs)
                if choice is not None:
                    idx, slot, row, col, plant_type, reason = choice
                    if idx >= INTERACT_BASE:
                        exec_code, exec_msg = client.execute_action(
                            ACTION_INTERACT_CELL,
                            row=row,
                            col=col,
                        )
                        if exec_code != 0:
                            return finish(
                                False,
                                "ACTION_ERROR",
                                f"mask[{idx}]=1 but InteractCell failed: {exec_code} {exec_msg}",
                            )
                        plants_executed += 1
                        _emit(
                            progress,
                            phases,
                            "INTERACT",
                            f"#{plants_executed} idx={idx} r{row}c{col} reason={reason}",
                        )
                    else:
                        before_ids = {int(p.get("id", -1)) for p in obs.get("plants", [])}
                        before_sun = int(obs.get("global", {}).get("sun", 0) or 0)
                        exec_code, exec_msg = client.execute_action(
                            ACTION_PLANT,
                            card_slot=slot,
                            row=row,
                            col=col,
                        )
                        if exec_code != 0:
                            return finish(
                                False,
                                "ACTION_ERROR",
                                f"mask[{idx}]=1 but Plant failed: {exec_code} {exec_msg}",
                            )
                        after, msg = _observation(client)
                        if not after:
                            return finish(False, "OBS_ERROR", f"post-Plant observation unavailable: {msg}")
                        after_plants = after.get("plants", [])
                        new_plant = any(int(p.get("id", -1)) not in before_ids for p in after_plants)
                        target_present = any(
                            int(p.get("row", -1)) == row and int(p.get("col", -1)) == col
                            for p in after_plants
                        )
                        if not new_plant or not target_present:
                            return finish(
                                False,
                                "ACTION_VERIFY_ERROR",
                                f"Plant success but no new target plant at r{row}c{col}",
                            )
                        plants_executed += 1
                        after_sun = int(after.get("global", {}).get("sun", 0) or 0)
                        _emit(
                            progress,
                            phases,
                            "PLANT",
                            (
                                f"#{plants_executed} idx={idx} type={plant_type} slot={slot} "
                                f"r{row}c{col} reason={reason}, sun {before_sun}->{after_sun}"
                            ),
                        )

            before_clock = int(obs.get("global", {}).get("game_clock", 0) or 0)
            if simulation_mode == SIMULATION_HEADLESS:
                stepped = int(client.dll.pvz_gym_advance_ticks_headless(step_ticks))
            elif simulation_mode in (SIMULATION_ANIMATED, SIMULATION_CONTINUOUS):
                target_clock = before_clock + step_ticks
                observed_clock = before_clock
                while observed_clock < target_clock:
                    if stop_event is not None and stop_event.is_set():
                        break
                    if pause_event is not None and not pause_event.is_set():
                        if not client.is_paused():
                            client.pause()
                        while not pause_event.is_set():
                            if stop_event is not None and stop_event.is_set():
                                break
                            time.sleep(0.05)
                        if stop_event is not None and stop_event.is_set():
                            break
                        if client.is_paused():
                            client.resume()
                    tick_state = _runtime(client)
                    if _terminal_outcome(tick_state) or str(tick_state.get("phase_name")) != PHASE_BATTLE:
                        break
                    observed_clock = int(client.get_game_clock())
                    time.sleep(0)
                stepped = max(0, observed_clock - before_clock)
            else:
                stepped = int(client.advance_ticks(step_ticks))
            if stepped <= 0:
                if stop_event is not None and stop_event.is_set():
                    return finish(False, "ABORTED", "用户终止")
                if _terminal_outcome(_runtime(client)):
                    continue
                on, _ = client.is_game_on()
                if not on:
                    if stop_event is not None:
                        stop_event.set()
                    return finish(False, "ABORTED", "游戏异常退出，自动触发终止")
                return finish(False, "CLOCK_ERROR", f"advance_ticks({step_ticks}) returned {stepped}")
            ticks_advanced += stepped

            post = _runtime(client)
            if post.get("error"):
                return finish(False, "API_ERROR", str(post.get("error")))
            if _terminal_outcome(post):
                continue

            after_tick, tick_msg = _observation(client)
            if not after_tick:
                on, _ = client.is_game_on()
                if not on:
                    if stop_event is not None:
                        stop_event.set()
                    return finish(False, "ABORTED", "游戏异常退出，自动触发终止")
                return finish(False, "OBS_ERROR", f"post-tick observation unavailable: {tick_msg}")
            after_clock = int(after_tick.get("global", {}).get("game_clock", 0) or 0)
            if after_clock <= before_clock:
                return finish(False, "CLOCK_ERROR", f"game_clock did not advance: {before_clock}->{after_clock}")

            if ticks_advanced >= next_progress_tick:
                gl = after_tick.get("global", {})
                _emit(
                    progress,
                    phases,
                    "PROGRESS",
                    (
                        f"ticks={ticks_advanced}/{max_ticks}, plants={plants_executed}, "
                        f"wave={gl.get('current_wave')}/{gl.get('total_waves')}, "
                        f"zombies={len(after_tick.get('zombies', []))}"
                    ),
                )
                while next_progress_tick <= ticks_advanced:
                    next_progress_tick += 1000

        return finish(False, "TIMEOUT", f"max_ticks={max_ticks} reached without WON/LOST")

    except Exception as exc:
        try:
            on, _ = client.is_game_on()
            if not on:
                if stop_event is not None:
                    stop_event.set()
                return finish(False, "ABORTED", f"游戏异常退出 ({exc})，自动触发终止")
        except Exception:
            pass
        return finish(False, "EXCEPTION", f"{spec.label} LIVE E2E exception: {exc}")
    finally:
        try:
            if headless_active:
                client.dll.pvz_gym_end_headless_mode()
        except Exception:
            pass
        try:
            if original_frame_ms > 0 and simulation_mode == SIMULATION_ANIMATED:
                client.dll.pvz_gym_set_frame_duration(original_frame_ms)
        except Exception:
            pass
        try:
            state = _runtime(client)
            if str(state.get("phase_name")) == PHASE_BATTLE:
                if original_paused:
                    if not client.is_paused():
                        client.pause()
                elif client.is_paused():
                    client.resume()
        except Exception:
            pass
        try:
            if old_automation is not None:
                client.set_automation_config(
                    bool(old_automation.get("enabled", True)),
                    bool(old_automation.get("auto_dialogue", True)),
                    bool(old_automation.get("auto_collect", True)),
                )
        except Exception:
            pass


def run_early_level_stress(
    client: Any,
    spec: EarlyLevelSpec,
    *,
    episodes: int = 5,
    expected_path: str = PLAYTHROUGH_AUTO,
    expected_playthrough: Optional[str] = None,
    progress: ProgressFn = None,
    max_ticks: int = 60000,
    step_ticks: int = 50,
    simulation_mode: str = SIMULATION_STEPPED,
    animated_frame_ms: int = 1,
    pause_event: Optional[threading.Event] = None,
    stop_event: Optional[threading.Event] = None,
    seed_priority=None,
    seed_selection_mode="adaptive",
) -> EarlyLevelStressResult:
    if expected_playthrough is not None:
        expected_path = expected_playthrough
    expected_path = _normalize_playthrough(expected_path)
    requested = max(1, int(episodes))
    t0 = time.perf_counter()
    results: List[EarlyLevelE2EResult] = []
    wins = losses = failures = path_mismatches = timeouts = 0

    for episode in range(1, requested + 1):
        if stop_event is not None and stop_event.is_set():
            break
        if progress:
            progress("STRESS", f"{spec.label} {expected_path} episode {episode}/{requested} begin")

        def ep_progress(phase: str, message: str, ep: int = episode) -> None:
            if progress:
                progress(f"E{ep}:{phase}", message)

        result = run_early_level_e2e(
            client,
            spec,
            expected_path=expected_path,
            progress=ep_progress,
            max_ticks=max_ticks,
            step_ticks=step_ticks,
            simulation_mode=simulation_mode,
            animated_frame_ms=animated_frame_ms,
            pause_event=pause_event,
            stop_event=stop_event,
            seed_priority=seed_priority,
            seed_selection_mode=seed_selection_mode,
        )
        results.append(result)
        if result.outcome == "ABORTED":
            if stop_event is not None and not stop_event.is_set():
                stop_event.set()
            break
        if result.success:
            if result.outcome == "WON":
                wins += 1
            elif result.outcome == "LOST":
                losses += 1
        else:
            failures += 1
            if result.outcome == "PATH_MISMATCH":
                path_mismatches += 1
            if result.outcome in {"TIMEOUT", "CLOCK_ERROR"}:
                timeouts += 1
        if not result.success and result.outcome in {
            "ABORTED",
            "NOT_STARTED",
            "API_ERROR",
            "PATH_MISMATCH",
            "PLAYTHROUGH_UNKNOWN",
        }:
            break

    completed = sum(1 for r in results if r.success)
    return EarlyLevelStressResult(
        success=(completed == requested and failures == 0),
        level=spec.label,
        expected_path=expected_path,
        requested_episodes=requested,
        completed_episodes=completed,
        wins=wins,
        losses=losses,
        failures=failures,
        path_mismatches=path_mismatches,
        timeouts=timeouts,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        episode_results=results,
    )


# 1-2 public wrappers.
def run_level_1_2_e2e(client: Any, **kwargs) -> EarlyLevelE2EResult:
    return run_early_level_e2e(client, LEVEL_1_2_SPEC, **kwargs)


def run_level_1_2_adventure1_e2e(client: Any, **kwargs) -> EarlyLevelE2EResult:
    return run_early_level_e2e(client, LEVEL_1_2_SPEC, expected_path=PLAYTHROUGH_1, **kwargs)


def run_level_1_2_adventure2_e2e(client: Any, **kwargs) -> EarlyLevelE2EResult:
    return run_early_level_e2e(client, LEVEL_1_2_SPEC, expected_path=PLAYTHROUGH_2, **kwargs)


def run_level_1_2_adventure1_stress(client: Any, **kwargs) -> EarlyLevelStressResult:
    return run_early_level_stress(client, LEVEL_1_2_SPEC, expected_path=PLAYTHROUGH_1, **kwargs)


def run_level_1_2_adventure2_stress(client: Any, **kwargs) -> EarlyLevelStressResult:
    return run_early_level_stress(client, LEVEL_1_2_SPEC, expected_path=PLAYTHROUGH_2, **kwargs)

# Compatibility aliases.
run_level_1_2_first_run_e2e = run_level_1_2_adventure1_e2e
run_level_1_2_replay_e2e = run_level_1_2_adventure2_e2e
run_level_1_2_stress = run_level_1_2_adventure2_stress


# 1-3 public wrappers.
def run_level_1_3_e2e(client: Any, **kwargs) -> EarlyLevelE2EResult:
    return run_early_level_e2e(client, LEVEL_1_3_SPEC, **kwargs)


def run_level_1_3_adventure1_e2e(client: Any, **kwargs) -> EarlyLevelE2EResult:
    return run_early_level_e2e(client, LEVEL_1_3_SPEC, expected_path=PLAYTHROUGH_1, **kwargs)


def run_level_1_3_adventure2_e2e(client: Any, **kwargs) -> EarlyLevelE2EResult:
    return run_early_level_e2e(client, LEVEL_1_3_SPEC, expected_path=PLAYTHROUGH_2, **kwargs)


def run_level_1_3_adventure1_stress(client: Any, **kwargs) -> EarlyLevelStressResult:
    return run_early_level_stress(client, LEVEL_1_3_SPEC, expected_path=PLAYTHROUGH_1, **kwargs)


def run_level_1_3_adventure2_stress(client: Any, **kwargs) -> EarlyLevelStressResult:
    return run_early_level_stress(client, LEVEL_1_3_SPEC, expected_path=PLAYTHROUGH_2, **kwargs)

# Compatibility aliases.
run_level_1_3_first_run_e2e = run_level_1_3_adventure1_e2e
run_level_1_3_replay_e2e = run_level_1_3_adventure2_e2e
run_level_1_3_stress = run_level_1_3_adventure2_stress


def install_early_levels_live_tests_gui(gui_cls: Any, ui: Any) -> None:
    """Add explicit Adventure-1 / Adventure-2 controls for 1-2 and 1-3."""
    if getattr(gui_cls, "_early_levels_live_tests_gui_installed", False):
        return

    original_build_left_panel = gui_cls._build_left_panel

    def _scroll_frame(parent):
        canvas = next((w for w in parent.winfo_children() if isinstance(w, tk.Canvas)), None)
        if canvas is None:
            return None
        return next((w for w in canvas.winfo_children() if isinstance(w, tk.Frame)), None)

    def _state(self, spec: EarlyLevelSpec) -> dict:
        states = getattr(self, "_early_level_live_states", None)
        if states is None:
            states = {}
            self._early_level_live_states = states
        return states.setdefault(spec.label, {"running": False, "pause": None, "stop": None})

    def _status_label(self, spec: EarlyLevelSpec):
        return getattr(self, f"lbl_early_{spec.level_id}_status", None)

    def _start(self, spec: EarlyLevelSpec, playthrough: str, episodes: int) -> None:
        state = _state(self, spec)
        if state["running"]:
            self.log("WARN", f"[{spec.label} LIVE] 已有实机测试正在运行")
            return
        state["running"] = True
        state["pause"] = threading.Event()
        state["pause"].set()
        state["stop"] = threading.Event()
        label = _status_label(self, spec)
        mode_text = "一周目" if playthrough == PLAYTHROUGH_1 else "二周目"
        if label is not None:
            label.config(
                text=f"{mode_text} {'E2E' if episodes == 1 else f'x{episodes}'}: RUNNING",
                fg=ui.COLOR_GOLD,
            )
        self.log(
            "HEADER",
            f"\n========== {spec.label} {playthrough} {'E2E' if episodes == 1 else f'STRESS x{episodes}'} =========="
        )

        def progress(phase: str, message: str):
            self.root.after(0, lambda: self.log("INFO", f"[{spec.label}:{phase}] {message}"))

        def worker():
            if episodes == 1:
                result = run_early_level_e2e(
                    self.client,
                    spec,
                    expected_path=playthrough,
                    progress=progress,
                    pause_event=state["pause"],
                    stop_event=state["stop"],
                )
            else:
                result = run_early_level_stress(
                    self.client,
                    spec,
                    episodes=episodes,
                    expected_path=playthrough,
                    progress=progress,
                    pause_event=state["pause"],
                    stop_event=state["stop"],
                )

            def finish_ui():
                state["running"] = False
                state["pause"] = None
                state["stop"] = None
                status = _status_label(self, spec)
                if episodes == 1:
                    if result.success:
                        self.log("PASS", f"{result.message} | {result.elapsed_ms:.2f}ms")
                        text = f"PASS | {result.actual_path} -> {result.outcome}"
                        color = ui.COLOR_GREEN
                    else:
                        aborted = result.outcome == "ABORTED"
                        self.log("WARN" if aborted else "FAIL", f"{result.outcome}: {result.message}")
                        text = f"{'ABORTED' if aborted else 'FAIL'} | {result.outcome}"
                        color = ui.COLOR_GOLD if aborted else ui.COLOR_RED
                else:
                    if result.success:
                        self.log(
                            "PASS",
                            f"{spec.label} {playthrough} stress PASS: "
                            f"{result.completed_episodes}/{result.requested_episodes}, "
                            f"W/L={result.wins}/{result.losses}, win_rate={result.win_rate:.1%}",
                        )
                        text = f"{mode_text} x{episodes} PASS | W/L {result.wins}/{result.losses}"
                        color = ui.COLOR_GREEN
                    else:
                        aborted = any(r.outcome == "ABORTED" for r in result.episode_results)
                        self.log(
                            "WARN" if aborted and result.failures == 0 else "FAIL",
                            f"{spec.label} {playthrough} stress {'ABORTED' if aborted else 'FAIL'}: "
                            f"completed={result.completed_episodes}/{result.requested_episodes}, "
                            f"failures={result.failures}, path_mismatch={result.path_mismatches}",
                        )
                        text = f"{mode_text} x{episodes} {'ABORTED' if aborted else 'FAIL'}"
                        color = ui.COLOR_GOLD if aborted and result.failures == 0 else ui.COLOR_RED
                if status is not None:
                    status.config(text=text, fg=color)
                self.refresh_status()

            self.root.after(0, finish_ui)

        threading.Thread(target=worker, daemon=True).start()

    def _toggle_pause(self, spec: EarlyLevelSpec) -> None:
        state = _state(self, spec)
        ev = state.get("pause")
        if ev is None:
            self.log("WARN", f"[{spec.label} LIVE] 当前没有运行中的测试")
            return
        if ev.is_set():
            ev.clear()
            self.log("WARN", f"[{spec.label} LIVE] 已暂停")
        else:
            ev.set()
            self.log("INFO", f"[{spec.label} LIVE] 已继续")

    def _stop(self, spec: EarlyLevelSpec) -> None:
        state = _state(self, spec)
        stop_ev = state.get("stop")
        pause_ev = state.get("pause")
        if stop_ev is None:
            self.log("WARN", f"[{spec.label} LIVE] 当前没有运行中的测试")
            return
        stop_ev.set()
        if pause_ev is not None:
            pause_ev.set()
        self.log("WARN", f"[{spec.label} LIVE] 终止信号已发送")

    def _build_card(self, frame, spec: EarlyLevelSpec) -> None:
        card = self._create_card(frame, f"{spec.label} LIVE：一周目 / 二周目")
        card.pack(fill=tk.X, pady=(0, 10))
        tk.Label(
            card,
            text=(
                "周目判定读取原版 PlayerInfo::mFinishedAdventure：0=一周目，>=1=二周目及以后。\n"
                "一周目使用该关历史早期卡组；二周目保留 Dave 强制卡，再用完整已解锁卡池补齐。"
            ),
            font=("Microsoft YaHei UI", 8),
            fg=ui.CLR_TEXT_MUTED,
            bg=ui.CLR_PANEL,
            justify="left",
            wraplength=700,
        ).pack(anchor="w", padx=12, pady=(0, 8))

        status = tk.Label(
            card,
            text="READY | 一周目 / 二周目",
            font=ui.FONT_BOLD,
            fg=ui.COLOR_BLUE,
            bg=ui.CLR_PANEL,
        )
        setattr(self, f"lbl_early_{spec.level_id}_status", status)
        status.pack(anchor="w", padx=12, pady=(0, 8))

        buttons = tk.Frame(card, bg=ui.CLR_PANEL)
        buttons.pack(fill=tk.X, padx=12, pady=(0, 4))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        ui.CyberButton(
            buttons,
            "[一周目 E2E]",
            lambda sp=spec: _start(self, sp, PLAYTHROUGH_1, 1),
            bg_color=ui.COLOR_GOLD,
            fg_color="#ffffff",
            height=36,
        ).grid(row=0, column=0, padx=(0, 4), sticky="ew")
        ui.CyberButton(
            buttons,
            "[二周目 E2E]",
            lambda sp=spec: _start(self, sp, PLAYTHROUGH_2, 1),
            bg_color=ui.COLOR_GOLD,
            fg_color="#ffffff",
            height=36,
        ).grid(row=0, column=1, padx=(4, 0), sticky="ew")

        stress = tk.Frame(card, bg=ui.CLR_PANEL)
        stress.pack(fill=tk.X, padx=12, pady=(0, 4))
        stress.columnconfigure(0, weight=1)
        stress.columnconfigure(1, weight=1)
        ui.CyberButton(
            stress,
            "[一周目 x5]",
            lambda sp=spec: _start(self, sp, PLAYTHROUGH_1, 5),
            bg_color=ui.CLR_PANEL,
            fg_color=ui.COLOR_GOLD,
            height=32,
        ).grid(row=0, column=0, padx=(0, 4), sticky="ew")
        ui.CyberButton(
            stress,
            "[二周目 x5]",
            lambda sp=spec: _start(self, sp, PLAYTHROUGH_2, 5),
            bg_color=ui.CLR_PANEL,
            fg_color=ui.COLOR_GOLD,
            height=32,
        ).grid(row=0, column=1, padx=(4, 0), sticky="ew")

        ctrl = tk.Frame(card, bg=ui.CLR_PANEL)
        ctrl.pack(fill=tk.X, padx=12, pady=(0, 12))
        ctrl.columnconfigure(0, weight=1)
        ctrl.columnconfigure(1, weight=1)
        ui.CyberButton(
            ctrl,
            "[暂停 / 继续]",
            lambda sp=spec: _toggle_pause(self, sp),
            bg_color=ui.CLR_PANEL,
            fg_color=ui.COLOR_GOLD,
            height=30,
        ).grid(row=0, column=0, padx=(0, 4), sticky="ew")
        ui.CyberButton(
            ctrl,
            "[终止]",
            lambda sp=spec: _stop(self, sp),
            bg_color=ui.CLR_PANEL,
            fg_color=ui.COLOR_RED,
            height=30,
        ).grid(row=0, column=1, padx=(4, 0), sticky="ew")

    def build_with_early_levels(self, parent):
        original_build_left_panel(self, parent)
        frame = _scroll_frame(parent)
        if frame is None:
            return
        _build_card(self, frame, LEVEL_1_2_SPEC)
        _build_card(self, frame, LEVEL_1_3_SPEC)

    gui_cls._build_left_panel = build_with_early_levels
    gui_cls._early_levels_live_tests_gui_installed = True
