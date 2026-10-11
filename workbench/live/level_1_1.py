# -*- coding: utf-8 -*-
"""Destructive live tests for adventure 1-1.

This module deliberately stays outside ``NativeUnitTestSuite.get_all_tests()``.
Both entry points switch the real game to Adventure 1-1:

``run_level_1_1_e2e``
    Single-run LIVE E2E environment test.  It validates one complete lifecycle
    from level entry through seed selection/battle to WON/LOST.

``run_level_1_1_stress``
    Repeats the same single-run contract for multiple episodes and reports
    lifecycle stability separately from battle win rate.

A deterministic-ish baseline deck and battle policy are used to give the test a
reasonable chance of winning.  WIN is *not* required for an environment PASS:
reaching either authoritative terminal state is enough.
"""

from __future__ import annotations

import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .level_1_1_policy import (
    LEVEL_1_1_DECK_PRIORITY as POLICY_DECK_PRIORITY,
    choose_level_1_1_plant,
)


ACTION_PLANT = 1
ACTION_SPACE_SIZE = 757

SEED_CAPACITY = 64
SEED_DESELECT_BASE = SEED_CAPACITY * 2
SEED_CONFIRM_ACTION = SEED_CAPACITY * 3

PHASE_INTRO = "LEVEL_INTRO"
PHASE_DIALOGUE = "DIALOGUE"
PHASE_SEED = "SEED_SELECTION"
PHASE_BATTLE = "BATTLE"
PHASE_COMPLETE = "LEVEL_COMPLETE"
PHASE_AWARD = "AWARD"
PHASE_LOST = "LOST"

LEVEL_1_1_DECK_PRIORITY = list(POLICY_DECK_PRIORITY)

ProgressFn = Optional[Callable[[str, str], None]]


@dataclass
class Level11E2EResult:
    success: bool
    outcome: str
    message: str
    elapsed_ms: float
    entry_path: str = "UNKNOWN"
    ticks_advanced: int = 0
    plants_executed: int = 0
    collected_items: int = 0
    seed_actions: int = 0
    phases: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Level11StressResult:
    success: bool
    requested_episodes: int
    completed_episodes: int
    wins: int
    losses: int
    failures: int
    setup_failures: int
    timeouts: int
    connection_or_exception_failures: int
    elapsed_ms: float
    episode_results: List[Level11E2EResult] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        terminals = self.wins + self.losses
        return (self.wins / terminals) if terminals else 0.0


def _emit(progress: ProgressFn, phases: List[Dict[str, Any]], phase: str, message: str) -> None:
    phases.append({"phase": phase, "message": message})
    if progress:
        progress(phase, message)


def _runtime(client: Any) -> dict:
    state = client.get_runtime_state()
    if not isinstance(state, dict):
        return {"phase_name": "UNKNOWN", "error": "runtime state is not a dict"}
    return state


def _wait_phase(
    client: Any,
    accepted: Sequence[str],
    *,
    timeout_s: float,
    level_id: Optional[int] = None,
) -> dict:
    deadline = time.monotonic() + timeout_s
    accepted_set = set(accepted)
    last: dict = {}
    while time.monotonic() < deadline:
        last = _runtime(client)
        if last.get("error"):
            return last
        if str(last.get("phase_name")) == "OFFLINE" or not last.get("process_available", True):
            return last
        if level_id is not None and int(last.get("level_id", 0) or 0) != level_id:
            time.sleep(0.01)
            continue
        if str(last.get("phase_name", "UNKNOWN")) in accepted_set:
            return last
        time.sleep(0.01)
    return last


def _wait_prebattle_ready(
    client: Any,
    *,
    safety_timeout_s: float,
    level_id: Optional[int] = None,
) -> dict:
    """Wait for authoritative seed-selection or battle facts."""
    deadline = time.monotonic() + safety_timeout_s
    last: dict = {}
    while time.monotonic() < deadline:
        last = _runtime(client)
        if last.get("error"):
            return last
        if str(last.get("phase_name")) == "OFFLINE" or not last.get("process_available", True):
            return last
        if level_id is not None and int(last.get("level_id", 0) or 0) != level_id:
            time.sleep(0.01)
            continue

        phase = str(last.get("phase_name", "UNKNOWN"))
        if phase == PHASE_BATTLE:
            return last
        if phase == PHASE_SEED:
            n, _, seed_state = client.get_seed_selection_json()
            if n > 0 and isinstance(seed_state, dict) and bool(seed_state.get("present", False)):
                return last
        if phase not in (PHASE_INTRO, PHASE_DIALOGUE, PHASE_SEED):
            return last
        time.sleep(0.01)
    return last


def _observation(client: Any) -> Tuple[Optional[dict], str]:
    n, msg, obs = client.get_observation_json()
    if n <= 0 or not isinstance(obs, dict):
        return None, msg or "observation unavailable"
    return obs, "success"


def _seed_state(client: Any) -> dict:
    n, msg, data = client.get_seed_selection_json()
    if n <= 0 or not isinstance(data, dict):
        raise RuntimeError(f"seed selection state unavailable: {msg}")
    if not bool(data.get("present", False)):
        raise RuntimeError("phase=SEED_SELECTION but SeedSelectionState.present=false")
    return data


def _seed_mask(client: Any) -> List[bool]:
    mask = list(client.get_seed_selection_mask())
    expected = SEED_CONFIRM_ACTION + 1
    if len(mask) != expected:
        raise RuntimeError(f"seed action mask length={len(mask)}, expected={expected}")
    return mask


def _selectable_seed(mask: List[bool], state: dict, priority: Sequence[int]) -> Optional[int]:
    selected = {int(v) for v in state.get("selected_seeds", [])}
    seen = set()
    for plant_type in list(priority) + list(range(SEED_CAPACITY)):
        plant_type = int(plant_type)
        if plant_type in seen:
            continue
        seen.add(plant_type)
        if plant_type in selected:
            continue
        if 0 <= plant_type < SEED_CAPACITY and bool(mask[plant_type]):
            return plant_type
    return None


def _exercise_seed_roundtrip(
    client: Any,
    progress: ProgressFn,
    phases: List[Dict[str, Any]],
) -> int:
    """Exercise one legal Select/Deselect pair without changing the final deck."""
    state = _seed_state(client)
    mask = _seed_mask(client)
    probe = _selectable_seed(mask, state, LEVEL_1_1_DECK_PRIORITY)
    if probe is None:
        return 0

    before = int(state.get("selected_count", 0) or 0)
    code, msg = client.select_seed(probe)
    if code != 0:
        raise RuntimeError(f"mask allowed Select({probe}) but call failed: {code} {msg}")
    actions = 1

    selected = _seed_state(client)
    if int(selected.get("selected_count", 0) or 0) != before + 1:
        raise RuntimeError(f"Select({probe}) did not increase selected_count")
    if bool(selected.get("seeds_in_flight", False)):
        raise RuntimeError("Select returned while a seed is still in flight")
    _emit(progress, phases, "SEED", f"Select({probe}) passed")

    after_mask = _seed_mask(client)
    if not bool(after_mask[SEED_DESELECT_BASE + probe]):
        _emit(progress, phases, "SEED", f"Deselect({probe}) unavailable; keeping selected seed")
        return actions

    code, msg = client.deselect_seed(probe)
    if code != 0:
        raise RuntimeError(f"mask allowed Deselect({probe}) but call failed: {code} {msg}")
    actions += 1

    deselected = _seed_state(client)
    if int(deselected.get("selected_count", 0) or 0) != before:
        raise RuntimeError(f"Deselect({probe}) did not restore selected_count")
    _emit(progress, phases, "SEED", f"Deselect({probe}) passed")
    return actions


def _normalize_level_1_1_deck(
    client: Any,
    progress: ProgressFn,
    phases: List[Dict[str, Any]],
) -> int:
    """Prefer a stable lawn deck while respecting locked/Dave-selected seeds."""
    actions = 0
    preferred = set(LEVEL_1_1_DECK_PRIORITY)

    state = _seed_state(client)
    # 提取 Dave/locked 强制选中的植物（disabled=true），不允许撤掉
    disabled_types = {
        int(opt.get("plant_type", -1))
        for opt in state.get("options", [])
        if bool(opt.get("disabled", False))
    }
    for plant_type in list(state.get("selected_seeds", [])):
        plant_type = int(plant_type)
        if plant_type in preferred:
            continue
        # Dave 强制选中的植物不得撤掉
        if plant_type in disabled_types:
            continue
        mask = _seed_mask(client)
        deselect_idx = SEED_DESELECT_BASE + plant_type
        if 0 <= deselect_idx < len(mask) and bool(mask[deselect_idx]):
            code, msg = client.deselect_seed(plant_type)
            if code != 0:
                raise RuntimeError(
                    f"mask allowed Deselect({plant_type}) during deck normalization: {code} {msg}"
                )
            actions += 1

    for _ in range(SEED_CAPACITY + 4):
        state = _seed_state(client)
        if bool(state.get("can_confirm", False)):
            break
        mask = _seed_mask(client)
        plant_type = _selectable_seed(mask, state, LEVEL_1_1_DECK_PRIORITY)
        if plant_type is None:
            raise RuntimeError(
                "deck is not confirmable and no legal Select remains: "
                f"{state.get('selected_count')}/{state.get('required_count')}"
            )
        code, msg = client.select_seed(plant_type)
        if code != 0:
            raise RuntimeError(f"Select({plant_type}) failed while filling deck: {code} {msg}")
        actions += 1
    else:
        raise RuntimeError("seed fill loop exceeded safety bound")

    ready = _seed_state(client)
    selected = [int(v) for v in ready.get("selected_seeds", [])]
    _emit(
        progress,
        phases,
        "SEED",
        "policy deck: " + ",".join(map(str, selected)),
    )
    return actions


def _run_seed_selection(client: Any, progress: ProgressFn, phases: List[Dict[str, Any]]) -> int:
    """Validate seed actions, build the baseline deck, and confirm into battle."""
    actions = 0
    state = _seed_state(client)
    _emit(
        progress,
        phases,
        "SEED",
        f"chooser ready: selected={state.get('selected_count')}/{state.get('required_count')}",
    )

    actions += _exercise_seed_roundtrip(client, progress, phases)
    actions += _normalize_level_1_1_deck(client, progress, phases)

    ready = _seed_state(client)
    mask = _seed_mask(client)
    if not bool(ready.get("can_confirm", False)) or not bool(mask[SEED_CONFIRM_ACTION]):
        raise RuntimeError("deck state and confirm mask are not both ready")
    if bool(ready.get("seeds_in_flight", False)):
        raise RuntimeError("full deck still has seeds in flight")

    _emit(
        progress,
        phases,
        "SEED",
        f"confirming {ready.get('selected_count')}/{ready.get('required_count')} deck",
    )
    code, msg = client.confirm_seeds()
    if code != 0:
        raise RuntimeError(f"confirm_seeds failed: {code} {msg}")
    actions += 1

    state = _wait_phase(
        client,
        [PHASE_BATTLE, PHASE_COMPLETE, PHASE_AWARD, PHASE_LOST],
        timeout_s=15.0,
        level_id=1,
    )
    if str(state.get("phase_name")) != PHASE_BATTLE:
        raise RuntimeError(
            f"confirm did not reach BATTLE: {state.get('phase_name')} / {state.get('raw_scene_name')}"
        )
    _emit(progress, phases, "SEED", "confirm passed; RuntimePhase -> BATTLE")
    return actions


def _terminal_outcome(state: dict) -> str:
    phase = str(state.get("phase_name", "UNKNOWN"))
    game_status = int(state.get("game_status", 0) or 0)
    if phase in (PHASE_COMPLETE, PHASE_AWARD) or game_status == 4:
        return "WON"
    if phase == PHASE_LOST or game_status == 5:
        return "LOST"
    return ""


def run_level_1_1_e2e(
    client: Any,
    *,
    progress: ProgressFn = None,
    max_ticks: int = 60000,
    step_ticks: int = 50,
    max_plants: int = 40,
    pause_event: Optional[threading.Event] = None,
    stop_event: Optional[threading.Event] = None,
) -> Level11E2EResult:
    """Run one destructive LIVE E2E episode on Adventure 1-1."""
    t0 = time.perf_counter()
    phases: List[Dict[str, Any]] = []
    entry_path = "UNKNOWN"
    ticks_advanced = 0
    plants_executed = 0
    collected_items = 0
    seed_actions = 0
    old_automation: Optional[dict] = None
    paused_by_test = False

    def finish(success: bool, outcome: str, message: str) -> Level11E2EResult:
        return Level11E2EResult(
            success=success,
            outcome=outcome,
            message=message,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            entry_path=entry_path,
            ticks_advanced=ticks_advanced,
            plants_executed=plants_executed,
            collected_items=collected_items,
            seed_actions=seed_actions,
            phases=phases,
        )

    try:
        if step_ticks <= 0:
            return finish(False, "CONFIG_ERROR", f"step_ticks must be > 0, got {step_ticks}")
        if max_ticks <= 0:
            return finish(False, "CONFIG_ERROR", f"max_ticks must be > 0, got {max_ticks}")

        if not getattr(client, "is_loaded", False):
            return finish(False, "NOT_STARTED", "DLL 未加载")

        api_probe = _runtime(client)
        if api_probe.get("error"):
            return finish(False, "API_ERROR", str(api_probe.get("error")))

        on, _ = client.is_game_on()
        if not on:
            ok, msg = client.attach()
            _emit(progress, phases, "ATTACH", f"attach={ok}: {msg}")
            on, _ = client.is_game_on()
        if not on:
            if stop_event is not None:
                stop_event.set()
            return finish(False, "ABORTED", "未找到运行中的 PlantsVsZombies.exe，自动终止")

        old_automation = client.get_automation_config()
        client.set_automation_config(True, True, True)

        _emit(progress, phases, "LEVEL", "单次测试：切换到冒险模式 1-1")
        code, msg = client.select_level(1, 1)
        if code != 0:
            return finish(False, "LEVEL_ERROR", f"select_level(1,1) failed: {code} {msg}")

        state = _wait_phase(
            client,
            [PHASE_INTRO, PHASE_DIALOGUE, PHASE_SEED, PHASE_BATTLE],
            timeout_s=5.0,
            level_id=1,
        )
        if str(state.get("phase_name")) == "OFFLINE" or not state.get("process_available", True):
            if stop_event is not None:
                stop_event.set()
            return finish(False, "ABORTED", "游戏异常退出，自动触发终止")
        if state.get("error"):
            return finish(False, "PHASE_ERROR", str(state.get("error")))

        _emit(
            progress,
            phases,
            "PHASE",
            f"entered {state.get('phase_name')} / {state.get('raw_scene_name')}",
        )

        if str(state.get("phase_name")) in (PHASE_INTRO, PHASE_DIALOGUE):
            _emit(progress, phases, "PREBATTLE", "推进原版 intro/Dave，直到选卡或战斗")
            ok = bool(client.handle_prebattle())
            _emit(progress, phases, "PREBATTLE", f"handle_prebattle={ok}")
            state = _wait_prebattle_ready(
                client,
                safety_timeout_s=60.0,
                level_id=1,
            )
            if str(state.get("phase_name")) == "OFFLINE" or not state.get("process_available", True):
                if stop_event is not None:
                    stop_event.set()
                return finish(False, "ABORTED", "游戏异常退出，自动触发终止")

        phase = str(state.get("phase_name", "UNKNOWN"))
        if phase == PHASE_SEED:
            entry_path = "SEED_SELECTION"
            _emit(progress, phases, "PATH", "1-1 重玩路径：进入标准选卡")
            seed_actions = _run_seed_selection(client, progress, phases)
        elif phase == PHASE_BATTLE:
            entry_path = "TUTORIAL_DIRECT"
            _emit(progress, phases, "PATH", "1-1 首次教学路径：直接进入战斗")
        else:
            return finish(False, "PHASE_ERROR", f"prebattle ended in unexpected phase={phase}")

        # 可跳过存在的教学机问题
        # if entry_path == "TUTORIAL_DIRECT":
        #     message = "1-1 首次教学局：检测到教学机制，跳过环境测试，直接返回"
        #     _emit(progress, phases, "SKIP", message)
        #     return finish(True, "TUTORIAL_SKIPPED", message)

        obs, msg = _observation(client)
        if not obs:
            return finish(False, "OBS_ERROR", f"battle observation unavailable: {msg}")
        gl = obs.get("global", {})
        if int(gl.get("level_id", -1)) != 1:
            return finish(False, "LEVEL_ERROR", f"observation level_id={gl.get('level_id')}, expected 1")
        if not obs.get("cards"):
            return finish(False, "OBS_ERROR", "battle cards are empty")

        _emit(
            progress,
            phases,
            "BATTLE",
            (
                f"ready path={entry_path}, sun={gl.get('sun')}, cards={len(obs.get('cards', []))}; "
                "policy=balanced-1-1"
            ),
        )

        if not client.is_paused():
            paused_by_test = bool(client.pause())
        if not client.is_paused():
            return finish(False, "CLOCK_ERROR", "cannot pause battle")
        _emit(progress, phases, "CLOCK", "paused; only advance_ticks drives time")

        next_progress_tick = 1000

        while ticks_advanced < max_ticks:
            # 终止检查
            if stop_event is not None and stop_event.is_set():
                return finish(False, "ABORTED", "用户终止")
            # 暂停检查：阻塞直到 pause_event 被 set（继续）
            if pause_event is not None:
                while not pause_event.is_set():
                    if stop_event is not None and stop_event.is_set():
                        return finish(False, "ABORTED", "用户终止")
                    time.sleep(0.05)

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
                _emit(
                    progress,
                    phases,
                    "TERMINAL",
                    f"{outcome}: phase={phase}, raw={state.get('raw_scene_name')}",
                )
                if plants_executed <= 0:
                    return finish(False, outcome, "terminal reached without any successful Plant action")
                return finish(
                    True,
                    outcome,
                    (
                        f"1-1 single LIVE E2E PASS: path={entry_path}, terminal={outcome}, "
                        f"seed_actions={seed_actions}, plants={plants_executed}, "
                        f"ticks={ticks_advanced}, collected={collected_items}"
                    ),
                )

            if phase != PHASE_BATTLE:
                return finish(
                    False,
                    "STATE_ERROR",
                    f"unexpected battle-loop phase={phase}, raw={state.get('raw_scene_name')}",
                )

            obs, msg = _observation(client)
            if not obs:
                on, _ = client.is_game_on()
                if not on:
                    if stop_event is not None:
                        stop_event.set()
                    return finish(False, "ABORTED", "游戏异常退出，自动触发终止")
                return finish(False, "OBS_ERROR", f"battle observation unavailable: {msg}")

            # 统一 automation tick 同时处理教学状态归一化与自动收集。
            # 首次 1-1 的状态 1..3 因此不会再依赖 GUI 守护线程。
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

                choice = choose_level_1_1_plant(list(mask), obs)
                if choice is not None:
                    idx, slot, row, col, plant_type, reason = choice
                    before_ids = {int(p.get("id", -1)) for p in obs.get("plants", [])}
                    before_sun = int(obs.get("global", {}).get("sun", 0) or 0)

                    exec_code, exec_msg = client.execute_action(
                        ACTION_PLANT, card_slot=slot, row=row, col=col
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
                            f"Plant returned success but no new target plant at r{row}c{col}",
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
                if stepped < 0:
                    return finish(False, "CLOCK_ERROR", f"advance_ticks({step_ticks}) returned {stepped}")
                return finish(False, "CLOCK_ERROR", f"advance_ticks({step_ticks}) made no progress")
            ticks_advanced += stepped

            post_tick_state = _runtime(client)
            if post_tick_state.get("error"):
                return finish(False, "API_ERROR", str(post_tick_state.get("error")))
            if _terminal_outcome(post_tick_state):
                continue

            after_tick_obs, tick_msg = _observation(client)
            if not after_tick_obs:
                on, _ = client.is_game_on()
                if not on:
                    if stop_event is not None:
                        stop_event.set()
                    return finish(False, "ABORTED", "游戏异常退出，自动触发终止")
                return finish(False, "OBS_ERROR", f"post-tick observation unavailable: {tick_msg}")
            after_clock = int(after_tick_obs.get("global", {}).get("game_clock", 0) or 0)
            if after_clock <= before_clock:
                return finish(
                    False,
                    "CLOCK_ERROR",
                    f"game_clock did not advance: before={before_clock}, after={after_clock}",
                )

            if ticks_advanced >= next_progress_tick:
                gl = after_tick_obs.get("global", {})
                _emit(
                    progress,
                    phases,
                    "PROGRESS",
                    (
                        f"ticks={ticks_advanced}/{max_ticks}, plants={plants_executed}, "
                        f"wave={gl.get('current_wave')}/{gl.get('total_waves')}, "
                        f"zombies={len(after_tick_obs.get('zombies', []))}"
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
        return finish(False, "EXCEPTION", f"1-1 single LIVE E2E exception: {exc}")
    finally:
        try:
            if paused_by_test and client.is_paused():
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


def run_level_1_1_stress(
    client: Any,
    *,
    episodes: int = 5,
    progress: ProgressFn = None,
    max_ticks: int = 60000,
    step_ticks: int = 50,
    max_plants: int = 40,
    pause_event: Optional[threading.Event] = None,
    stop_event: Optional[threading.Event] = None,
) -> Level11StressResult:
    """Repeat the single-run 1-1 LIVE E2E contract."""
    t0 = time.perf_counter()
    requested = max(1, int(episodes))
    results: List[Level11E2EResult] = []
    wins = losses = failures = 0
    setup_failures = timeouts = connection_or_exception_failures = 0

    if progress:
        progress("STRESS", f"start episodes={requested}")

    for episode in range(1, requested + 1):
        if progress:
            progress("STRESS", f"episode {episode}/{requested} begin")

        def episode_progress(phase: str, message: str, ep: int = episode) -> None:
            if progress:
                progress(f"E{ep}:{phase}", message)

        # 集数开始前检查终止
        if stop_event is not None and stop_event.is_set():
            if progress:
                progress("STRESS", f"episode {episode}/{requested} aborted before start")
            break

        result = run_level_1_1_e2e(
            client,
            progress=episode_progress,
            max_ticks=max_ticks,
            step_ticks=step_ticks,
            max_plants=max_plants,
            pause_event=pause_event,
            stop_event=stop_event,
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
            if result.outcome in {"LEVEL_ERROR", "PHASE_ERROR", "STATE_ERROR"}:
                setup_failures += 1
            if result.outcome in {"TIMEOUT", "CLOCK_ERROR"}:
                timeouts += 1
            if result.outcome in {"NOT_STARTED", "API_ERROR", "EXCEPTION"}:
                connection_or_exception_failures += 1

        if progress:
            progress(
                "STRESS",
                (
                    f"episode {episode}/{requested}: "
                    f"{'PASS' if result.success else 'FAIL'} terminal={result.outcome} "
                    f"ticks={result.ticks_advanced} plants={result.plants_executed}"
                ),
            )

        if not result.success and result.outcome in {"NOT_STARTED", "API_ERROR"}:
            break

    completed = sum(1 for result in results if result.success)
    success = completed == requested and failures == 0
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if progress:
        progress(
            "STRESS",
            (
                f"done completed={completed}/{requested}, wins={wins}, losses={losses}, "
                f"failures={failures}, win_rate={wins / max(1, wins + losses):.1%}"
            ),
        )

    return Level11StressResult(
        success=success,
        requested_episodes=requested,
        completed_episodes=completed,
        wins=wins,
        losses=losses,
        failures=failures,
        setup_failures=setup_failures,
        timeouts=timeouts,
        connection_or_exception_failures=connection_or_exception_failures,
        elapsed_ms=elapsed_ms,
        episode_results=results,
    )


def install_level_1_1_live_tests_gui(gui_cls: Any, ui: Any) -> None:
    if getattr(gui_cls, "_level_1_1_live_tests_gui_installed", False):
        return

    original_build_left_panel = gui_cls._build_left_panel

    def _scroll_frame(parent):
        canvas = next((w for w in parent.winfo_children() if isinstance(w, tk.Canvas)), None)
        if canvas is None:
            return None
        return next((w for w in canvas.winfo_children() if isinstance(w, tk.Frame)), None)

    def _set_running(self, running: bool) -> None:
        self._level_1_1_live_running = running

    def cmd_run_level_1_1_e2e(self):
        if getattr(self, "_level_1_1_live_running", False):
            self.log("WARN", "[1-1 LIVE] 另一个 1-1 实机测试正在运行")
            return

        _set_running(self, True)
        self._live_pause_event = threading.Event()
        self._live_pause_event.set()
        self._live_stop_event = threading.Event()

        if hasattr(self, "lbl_level_1_1_e2e_status"):
            self.lbl_level_1_1_e2e_status.config(text="单次: RUNNING", fg=ui.COLOR_GOLD)
        self.log("HEADER", "\n========== 1-1 SINGLE LIVE E2E ==========")
        self.log("WARN", "单次环境测试：会主动切换到 1-1；普通单元测试不会执行它")

        def progress(phase: str, message: str):
            self.root.after(0, lambda: self.log("INFO", f"[{phase}] {message}"))

        def worker():
            result = run_level_1_1_e2e(
                self.client,
                progress=progress,
                pause_event=self._live_pause_event,
                stop_event=self._live_stop_event,
            )

            def finish_ui():
                _set_running(self, False)
                self._live_pause_event = None
                self._live_stop_event = None
                if result.success:
                    self.log(
                        "PASS",
                        f"{result.message} | 总耗时: {result.elapsed_ms:.2f}ms",
                    )
                    color = ui.COLOR_GREEN
                    text = f"单次: PASS | {result.entry_path} -> {result.outcome}"
                else:
                    aborted = result.outcome == "ABORTED"
                    level = "WARN" if aborted else "FAIL"
                    self.log(
                        level,
                        (
                            f"{result.outcome}: {result.message} | "
                            f"总耗时: {result.elapsed_ms:.2f}ms"
                        ),
                    )
                    color = ui.COLOR_GOLD if aborted else ui.COLOR_RED
                    text = f"单次: {'ABORTED' if aborted else 'FAIL'} | {result.outcome}"
                if hasattr(self, "lbl_level_1_1_e2e_status"):
                    self.lbl_level_1_1_e2e_status.config(text=text, fg=color)
                self.refresh_status()

            self.root.after(0, finish_ui)

        threading.Thread(target=worker, daemon=True).start()

    def cmd_run_level_1_1_stress(self):
        if getattr(self, "_level_1_1_live_running", False):
            self.log("WARN", "[1-1 LIVE] 另一个 1-1 实机测试正在运行")
            return

        _set_running(self, True)
        # 初始化控制 Event：pause_event set=运行中，stop_event clear=未终止
        self._live_pause_event = threading.Event()
        self._live_pause_event.set()   # 默认不暂停
        self._live_stop_event = threading.Event()

        if hasattr(self, "lbl_level_1_1_stress_status"):
            self.lbl_level_1_1_stress_status.config(text="压力: RUNNING (5 episodes)", fg=ui.COLOR_GOLD)
        self.log("HEADER", "\n========== 1-1 LIVE STRESS x5 ==========")
        self.log("WARN", "压力测试会连续重开 1-1；胜负只计统计，环境失败才判 FAIL")

        def progress(phase: str, message: str):
            self.root.after(0, lambda: self.log("INFO", f"[{phase}] {message}"))

        def worker():
            result = run_level_1_1_stress(
                self.client,
                episodes=5,
                progress=progress,
                pause_event=self._live_pause_event,
                stop_event=self._live_stop_event,
            )

            def finish_ui():
                _set_running(self, False)
                # 清理 Event，避免遗留
                self._live_pause_event = None
                self._live_stop_event = None
                if result.success:
                    self.log(
                        "PASS",
                        (
                            f"1-1 stress PASS: completed={result.completed_episodes}/"
                            f"{result.requested_episodes}, W/L={result.wins}/{result.losses}, "
                            f"win_rate={result.win_rate:.1%}, 总耗时={result.elapsed_ms:.2f}ms"
                        ),
                    )
                    color = ui.COLOR_GREEN
                    text = (
                        f"压力: PASS | {result.completed_episodes}/{result.requested_episodes} "
                        f"| W/L {result.wins}/{result.losses}"
                    )
                else:
                    # 用户主动终止时用 WARN 级别，而非 FAIL
                    aborted = any(
                        r.outcome == "ABORTED" for r in result.episode_results
                    ) or result.completed_episodes < result.requested_episodes
                    level = "WARN" if aborted and result.failures == 0 else "FAIL"
                    self.log(
                        level,
                        (
                            f"1-1 stress {'ABORTED' if aborted else 'FAIL'}: "
                            f"completed={result.completed_episodes}/"
                            f"{result.requested_episodes}, failures={result.failures}, "
                            f"setup={result.setup_failures}, timeouts={result.timeouts}, "
                            f"总耗时={result.elapsed_ms:.2f}ms"
                        ),
                    )
                    color = ui.COLOR_GOLD if aborted and result.failures == 0 else ui.COLOR_RED
                    text = (
                        f"压力: {'ABORTED' if aborted else 'FAIL'} | "
                        f"{result.completed_episodes}/{result.requested_episodes} "
                        f"| failures={result.failures}"
                    )
                if hasattr(self, "lbl_level_1_1_stress_status"):
                    self.lbl_level_1_1_stress_status.config(text=text, fg=color)
                self.refresh_status()

            self.root.after(0, finish_ui)

        threading.Thread(target=worker, daemon=True).start()

    def cmd_toggle_live_pause(self):
        """暂停 / 继续：切换 pause_event 的 set/clear 状态。"""
        ev: Optional[threading.Event] = getattr(self, "_live_pause_event", None)
        if ev is None:
            self.log("WARN", "[1-1 LIVE] 当前没有正在运行的实机测试")
            return
        if ev.is_set():
            ev.clear()   # 暂停
            self.log("WARN", "[1-1 LIVE] 已暂停（当前步推进后生效）")
            if hasattr(self, "lbl_level_1_1_stress_status") and "RUNNING" in self.lbl_level_1_1_stress_status.cget("text"):
                self.lbl_level_1_1_stress_status.config(text="压力: PAUSED", fg=ui.COLOR_GOLD)
            if hasattr(self, "lbl_level_1_1_e2e_status") and "RUNNING" in self.lbl_level_1_1_e2e_status.cget("text"):
                self.lbl_level_1_1_e2e_status.config(text="单次: PAUSED", fg=ui.COLOR_GOLD)
        else:
            ev.set()     # 继续
            self.log("INFO", "[1-1 LIVE] 已继续运行")
            if hasattr(self, "lbl_level_1_1_stress_status") and "PAUSED" in self.lbl_level_1_1_stress_status.cget("text"):
                self.lbl_level_1_1_stress_status.config(text="压力: RUNNING", fg=ui.COLOR_GOLD)
            if hasattr(self, "lbl_level_1_1_e2e_status") and "PAUSED" in self.lbl_level_1_1_e2e_status.cget("text"):
                self.lbl_level_1_1_e2e_status.config(text="单次: RUNNING", fg=ui.COLOR_GOLD)

    def cmd_stop_live(self):
        """终止：设置 stop_event，让当前测试在下一个检查点退出。"""
        stop_ev: Optional[threading.Event] = getattr(self, "_live_stop_event", None)
        pause_ev: Optional[threading.Event] = getattr(self, "_live_pause_event", None)
        if stop_ev is None:
            self.log("WARN", "[1-1 LIVE] 当前没有正在运行的实机测试")
            return
        stop_ev.set()
        # 若处于暂停状态，先恢复让循环检测到 stop_event
        if pause_ev is not None and not pause_ev.is_set():
            pause_ev.set()
        self.log("WARN", "[1-1 LIVE] 终止信号已发送，等待当前步安全退出...")
        if hasattr(self, "lbl_level_1_1_stress_status") and "RUNNING" in self.lbl_level_1_1_stress_status.cget("text"):
            self.lbl_level_1_1_stress_status.config(text="压力: STOPPING...", fg=ui.COLOR_RED)
        if hasattr(self, "lbl_level_1_1_e2e_status") and "RUNNING" in self.lbl_level_1_1_e2e_status.cget("text"):
            self.lbl_level_1_1_e2e_status.config(text="单次: STOPPING...", fg=ui.COLOR_RED)

    def build_with_live_tests(self, parent):
        original_build_left_panel(self, parent)
        frame = _scroll_frame(parent)
        if frame is None:
            return

        card = self._create_card(frame, "1-1 LIVE 环境测试（会切关）")
        card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            card,
            text=(
                "单次 E2E：完整跑 1 局，验证 RuntimePhase、选卡、Observation、Action Mask、"
                "真实种植、自动收集、Tick 推进与 WON/LOST 终局。LOST 也可环境 PASS。\n"
                "压力测试：连续跑 5 局相同 contract，统计稳定性与 W/L；"
                "使用固定优先选卡 + balanced-1-1 baseline，尽量提高通关概率。"
            ),
            font=("Microsoft YaHei UI", 8),
            fg=ui.CLR_TEXT_MUTED,
            bg=ui.CLR_PANEL,
            justify="left",
            wraplength=700,
        ).pack(anchor="w", padx=12, pady=(0, 8))

        self.lbl_level_1_1_e2e_status = tk.Label(
            card,
            text="单次: READY | 普通“运行全部测试”不会触发",
            font=ui.FONT_BOLD,
            fg=ui.COLOR_BLUE,
            bg=ui.CLR_PANEL,
        )
        self.lbl_level_1_1_e2e_status.pack(anchor="w", padx=12, pady=(0, 4))

        self.lbl_level_1_1_stress_status = tk.Label(
            card,
            text="压力: READY | 5 episodes",
            font=ui.FONT_BOLD,
            fg=ui.COLOR_BLUE,
            bg=ui.CLR_PANEL,
        )
        self.lbl_level_1_1_stress_status.pack(anchor="w", padx=12, pady=(0, 8))

        buttons = tk.Frame(card, bg=ui.CLR_PANEL)
        buttons.pack(fill=tk.X, padx=12, pady=(0, 4))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)

        ui.CyberButton(
            buttons,
            "[运行单次 1-1 E2E]",
            self.cmd_run_level_1_1_e2e,
            bg_color=ui.COLOR_GOLD,
            fg_color="#ffffff",
            height=36,
        ).grid(row=0, column=0, padx=(0, 4), sticky="ew")

        ui.CyberButton(
            buttons,
            "[运行 1-1 压力测试 x5]",
            self.cmd_run_level_1_1_stress,
            bg_color=ui.COLOR_GOLD,
            fg_color="#ffffff",
            height=36,
        ).grid(row=0, column=1, padx=(4, 0), sticky="ew")

        # 第二行：压力测试运行时控制按钮
        ctrl = tk.Frame(card, bg=ui.CLR_PANEL)
        ctrl.pack(fill=tk.X, padx=12, pady=(0, 12))
        ctrl.columnconfigure(0, weight=1)
        ctrl.columnconfigure(1, weight=1)

        ui.CyberButton(
            ctrl,
            "[暂停 / 继续]",
            self.cmd_toggle_live_pause,
            bg_color=ui.CLR_PANEL,
            fg_color=ui.COLOR_GOLD,
            height=30,
        ).grid(row=0, column=0, padx=(0, 4), sticky="ew")

        ui.CyberButton(
            ctrl,
            "[终止]",
            self.cmd_stop_live,
            bg_color=ui.CLR_PANEL,
            fg_color=ui.COLOR_RED,
            height=30,
        ).grid(row=0, column=1, padx=(4, 0), sticky="ew")

    gui_cls.cmd_run_level_1_1_e2e = cmd_run_level_1_1_e2e
    gui_cls.cmd_run_level_1_1_stress = cmd_run_level_1_1_stress
    gui_cls.cmd_toggle_live_pause = cmd_toggle_live_pause
    gui_cls.cmd_stop_live = cmd_stop_live
    gui_cls._build_left_panel = build_with_live_tests
    gui_cls._level_1_1_live_tests_gui_installed = True


install_level_1_1_e2e_gui = install_level_1_1_live_tests_gui
