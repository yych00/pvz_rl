# -*- coding: utf-8 -*-
"""Destructive LIVE E2E/stress tests for Adventure 1-2.

This deliberately stays outside the normal unit-test suite because it switches
and advances the real game process.  It reuses the proven lifecycle helpers
from the 1-1 harness while keeping 1-2 deck/policy decisions separate.
"""

from __future__ import annotations

import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

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
from .level_1_2_policy import LEVEL_1_2_DECK_PRIORITY, choose_level_1_2_plant


ProgressFn = Optional[Callable[[str, str], None]]


@dataclass
class Level12E2EResult:
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
class Level12StressResult:
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
    episode_results: List[Level12E2EResult] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        terminals = self.wins + self.losses
        return self.wins / terminals if terminals else 0.0


def _selectable_seed(mask: Sequence[bool], state: dict) -> Optional[int]:
    selected = {int(v) for v in state.get("selected_seeds", [])}
    seen = set()
    for plant_type in list(LEVEL_1_2_DECK_PRIORITY) + list(range(SEED_CAPACITY)):
        plant_type = int(plant_type)
        if plant_type in seen or plant_type in selected:
            continue
        seen.add(plant_type)
        if 0 <= plant_type < SEED_CAPACITY and bool(mask[plant_type]):
            return plant_type
    return None


def _run_seed_selection_1_2(client: Any, progress: ProgressFn, phases: List[Dict[str, Any]]) -> int:
    """Build the smallest legal 1-2 deck and confirm into battle."""
    actions = 0
    state = _seed_state(client)
    preferred = set(LEVEL_1_2_DECK_PRIORITY)
    disabled_types = {
        int(opt.get("plant_type", -1))
        for opt in state.get("options", [])
        if bool(opt.get("disabled", False))
    }

    _emit(
        progress,
        phases,
        "SEED",
        f"1-2 chooser ready: selected={state.get('selected_count')}/{state.get('required_count')}",
    )

    # Remove optional unrelated cards while preserving Dave/locked choices.
    for plant_type in list(state.get("selected_seeds", [])):
        plant_type = int(plant_type)
        if plant_type in preferred or plant_type in disabled_types:
            continue
        mask = _seed_mask(client)
        idx = SEED_DESELECT_BASE + plant_type
        if 0 <= idx < len(mask) and bool(mask[idx]):
            code, msg = client.deselect_seed(plant_type)
            if code != 0:
                raise RuntimeError(f"Deselect({plant_type}) failed: {code} {msg}")
            actions += 1

    for _ in range(SEED_CAPACITY + 4):
        state = _seed_state(client)
        if bool(state.get("can_confirm", False)):
            break
        mask = _seed_mask(client)
        plant_type = _selectable_seed(mask, state)
        if plant_type is None:
            raise RuntimeError(
                "1-2 deck not confirmable and no legal Select remains: "
                f"{state.get('selected_count')}/{state.get('required_count')}"
            )
        code, msg = client.select_seed(plant_type)
        if code != 0:
            raise RuntimeError(f"Select({plant_type}) failed: {code} {msg}")
        actions += 1
    else:
        raise RuntimeError("1-2 seed fill exceeded safety bound")

    ready = _seed_state(client)
    mask = _seed_mask(client)
    if not bool(ready.get("can_confirm", False)) or not bool(mask[SEED_CONFIRM_ACTION]):
        raise RuntimeError("1-2 deck and confirm mask are not both ready")

    selected = [int(v) for v in ready.get("selected_seeds", [])]
    _emit(progress, phases, "SEED", "1-2 policy deck: " + ",".join(map(str, selected)))

    code, msg = client.confirm_seeds()
    if code != 0:
        raise RuntimeError(f"confirm_seeds failed: {code} {msg}")
    actions += 1

    state = _wait_phase(
        client,
        [PHASE_BATTLE, PHASE_COMPLETE, PHASE_AWARD, PHASE_LOST],
        timeout_s=15.0,
        level_id=2,
    )
    if str(state.get("phase_name")) != PHASE_BATTLE:
        raise RuntimeError(
            f"1-2 confirm did not reach BATTLE: {state.get('phase_name')} / {state.get('raw_scene_name')}"
        )
    _emit(progress, phases, "SEED", "1-2 confirm passed; RuntimePhase -> BATTLE")
    return actions


def run_level_1_2_e2e(
    client: Any,
    *,
    progress: ProgressFn = None,
    max_ticks: int = 60000,
    step_ticks: int = 50,
    max_plants: int = 35,
    pause_event: Optional[threading.Event] = None,
    stop_event: Optional[threading.Event] = None,
) -> Level12E2EResult:
    """Run one destructive LIVE E2E episode on Adventure 1-2."""
    t0 = time.perf_counter()
    phases: List[Dict[str, Any]] = []
    entry_path = "UNKNOWN"
    ticks_advanced = plants_executed = collected_items = seed_actions = 0
    old_automation: Optional[dict] = None
    paused_by_test = False

    def finish(success: bool, outcome: str, message: str) -> Level12E2EResult:
        return Level12E2EResult(
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
        if step_ticks <= 0 or max_ticks <= 0:
            return finish(False, "CONFIG_ERROR", "step_ticks/max_ticks must be > 0")
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

        old_automation = client.get_automation_config()
        client.set_automation_config(True, True, True)

        _emit(progress, phases, "LEVEL", "单次测试：切换到冒险模式 1-2")
        code, msg = client.select_level(1, 2)
        if code != 0:
            return finish(False, "LEVEL_ERROR", f"select_level(1,2) failed: {code} {msg}")

        state = _wait_phase(
            client,
            [PHASE_INTRO, PHASE_DIALOGUE, PHASE_SEED, PHASE_BATTLE],
            timeout_s=5.0,
            level_id=2,
        )
        if state.get("error"):
            return finish(False, "PHASE_ERROR", str(state.get("error")))

        if str(state.get("phase_name")) in (PHASE_INTRO, PHASE_DIALOGUE):
            _emit(progress, phases, "PREBATTLE", "推进 1-2 intro/Dave，直到选卡或战斗")
            client.handle_prebattle()
            state = _wait_prebattle_ready(client, safety_timeout_s=60.0, level_id=2)

        phase = str(state.get("phase_name", "UNKNOWN"))
        if phase == PHASE_SEED:
            entry_path = "SEED_SELECTION"
            seed_actions = _run_seed_selection_1_2(client, progress, phases)
        elif phase == PHASE_BATTLE:
            entry_path = "DIRECT_BATTLE"
            _emit(progress, phases, "PATH", "1-2 直接进入战斗")
        else:
            return finish(False, "PHASE_ERROR", f"prebattle ended in unexpected phase={phase}")

        obs, msg = _observation(client)
        if not obs:
            return finish(False, "OBS_ERROR", f"battle observation unavailable: {msg}")
        gl = obs.get("global", {})
        if int(gl.get("level_id", -1)) != 2:
            return finish(False, "LEVEL_ERROR", f"observation level_id={gl.get('level_id')}, expected 2")
        if not obs.get("cards"):
            return finish(False, "OBS_ERROR", "battle cards are empty")

        _emit(
            progress,
            phases,
            "BATTLE",
            f"ready path={entry_path}, sun={gl.get('sun')}, cards={len(obs.get('cards', []))}; policy=1-2",
        )

        if not client.is_paused():
            paused_by_test = bool(client.pause())
        if not client.is_paused():
            return finish(False, "CLOCK_ERROR", "cannot pause battle")

        next_progress_tick = 1000
        while ticks_advanced < max_ticks:
            if stop_event is not None and stop_event.is_set():
                return finish(False, "ABORTED", "用户终止")
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
                _emit(progress, phases, "TERMINAL", f"{outcome}: phase={phase}")
                if plants_executed <= 0:
                    return finish(False, outcome, "terminal reached without a successful Plant action")
                return finish(
                    True,
                    outcome,
                    f"1-2 LIVE E2E PASS: terminal={outcome}, plants={plants_executed}, ticks={ticks_advanced}",
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
                choice = choose_level_1_2_plant(list(mask), obs)
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
                        return finish(False, "ACTION_VERIFY_ERROR", f"no new target plant at r{row}c{col}")
                    plants_executed += 1
                    after_sun = int(after.get("global", {}).get("sun", 0) or 0)
                    _emit(
                        progress,
                        phases,
                        "PLANT",
                        f"#{plants_executed} idx={idx} type={plant_type} r{row}c{col} reason={reason}, sun {before_sun}->{after_sun}",
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
                    f"ticks={ticks_advanced}/{max_ticks}, plants={plants_executed}, wave={gl.get('current_wave')}/{gl.get('total_waves')}, zombies={len(after_tick.get('zombies', []))}",
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
        return finish(False, "EXCEPTION", f"1-2 LIVE E2E exception: {exc}")
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


def run_level_1_2_stress(
    client: Any,
    *,
    episodes: int = 5,
    progress: ProgressFn = None,
    max_ticks: int = 60000,
    step_ticks: int = 50,
    max_plants: int = 35,
    pause_event: Optional[threading.Event] = None,
    stop_event: Optional[threading.Event] = None,
) -> Level12StressResult:
    t0 = time.perf_counter()
    requested = max(1, int(episodes))
    results: List[Level12E2EResult] = []
    wins = losses = failures = 0
    setup_failures = timeouts = connection_or_exception_failures = 0

    for episode in range(1, requested + 1):
        if stop_event is not None and stop_event.is_set():
            break
        if progress:
            progress("STRESS", f"1-2 episode {episode}/{requested} begin")

        def episode_progress(phase: str, message: str, ep: int = episode) -> None:
            if progress:
                progress(f"E{ep}:{phase}", message)

        result = run_level_1_2_e2e(
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
        if not result.success and result.outcome in {"ABORTED", "NOT_STARTED", "API_ERROR"}:
            break

    completed = sum(1 for r in results if r.success)
    return Level12StressResult(
        success=(completed == requested and failures == 0),
        requested_episodes=requested,
        completed_episodes=completed,
        wins=wins,
        losses=losses,
        failures=failures,
        setup_failures=setup_failures,
        timeouts=timeouts,
        connection_or_exception_failures=connection_or_exception_failures,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        episode_results=results,
    )


def install_level_1_2_live_tests_gui(gui_cls: Any, ui: Any) -> None:
    if getattr(gui_cls, "_level_1_2_live_tests_gui_installed", False):
        return

    original_build_left_panel = gui_cls._build_left_panel

    def _scroll_frame(parent):
        canvas = next((w for w in parent.winfo_children() if isinstance(w, tk.Canvas)), None)
        if canvas is None:
            return None
        return next((w for w in canvas.winfo_children() if isinstance(w, tk.Frame)), None)

    def _set_running(self, running: bool) -> None:
        self._level_1_2_live_running = running

    def cmd_run_level_1_2_e2e(self):
        if getattr(self, "_level_1_2_live_running", False):
            self.log("WARN", "[1-2 LIVE] 另一个 1-2 实机测试正在运行")
            return
        _set_running(self, True)
        self._level_1_2_pause_event = threading.Event()
        self._level_1_2_pause_event.set()
        self._level_1_2_stop_event = threading.Event()
        self.lbl_level_1_2_e2e_status.config(text="单次: RUNNING", fg=ui.COLOR_GOLD)
        self.log("HEADER", "\n========== 1-2 SINGLE LIVE E2E ==========")

        def progress(phase: str, message: str):
            self.root.after(0, lambda: self.log("INFO", f"[1-2:{phase}] {message}"))

        def worker():
            result = run_level_1_2_e2e(
                self.client,
                progress=progress,
                pause_event=self._level_1_2_pause_event,
                stop_event=self._level_1_2_stop_event,
            )

            def finish_ui():
                _set_running(self, False)
                self._level_1_2_pause_event = None
                self._level_1_2_stop_event = None
                if result.success:
                    self.log("PASS", f"{result.message} | {result.elapsed_ms:.2f}ms")
                    self.lbl_level_1_2_e2e_status.config(
                        text=f"单次: PASS | {result.entry_path} -> {result.outcome}", fg=ui.COLOR_GREEN
                    )
                else:
                    aborted = result.outcome == "ABORTED"
                    self.log("WARN" if aborted else "FAIL", f"{result.outcome}: {result.message}")
                    self.lbl_level_1_2_e2e_status.config(
                        text=f"单次: {'ABORTED' if aborted else 'FAIL'} | {result.outcome}",
                        fg=ui.COLOR_GOLD if aborted else ui.COLOR_RED,
                    )
                self.refresh_status()

            self.root.after(0, finish_ui)

        threading.Thread(target=worker, daemon=True).start()

    def cmd_run_level_1_2_stress(self):
        if getattr(self, "_level_1_2_live_running", False):
            self.log("WARN", "[1-2 LIVE] 另一个 1-2 实机测试正在运行")
            return
        _set_running(self, True)
        self._level_1_2_pause_event = threading.Event()
        self._level_1_2_pause_event.set()
        self._level_1_2_stop_event = threading.Event()
        self.lbl_level_1_2_stress_status.config(text="压力: RUNNING (5 episodes)", fg=ui.COLOR_GOLD)
        self.log("HEADER", "\n========== 1-2 LIVE STRESS x5 ==========")

        def progress(phase: str, message: str):
            self.root.after(0, lambda: self.log("INFO", f"[1-2:{phase}] {message}"))

        def worker():
            result = run_level_1_2_stress(
                self.client,
                episodes=5,
                progress=progress,
                pause_event=self._level_1_2_pause_event,
                stop_event=self._level_1_2_stop_event,
            )

            def finish_ui():
                _set_running(self, False)
                self._level_1_2_pause_event = None
                self._level_1_2_stop_event = None
                if result.success:
                    self.log(
                        "PASS",
                        f"1-2 stress PASS: {result.completed_episodes}/{result.requested_episodes}, W/L={result.wins}/{result.losses}, win_rate={result.win_rate:.1%}",
                    )
                    self.lbl_level_1_2_stress_status.config(
                        text=f"压力: PASS | {result.completed_episodes}/{result.requested_episodes} | W/L {result.wins}/{result.losses}",
                        fg=ui.COLOR_GREEN,
                    )
                else:
                    aborted = any(r.outcome == "ABORTED" for r in result.episode_results)
                    self.log(
                        "WARN" if aborted and result.failures == 0 else "FAIL",
                        f"1-2 stress {'ABORTED' if aborted else 'FAIL'}: completed={result.completed_episodes}/{result.requested_episodes}, failures={result.failures}",
                    )
                    self.lbl_level_1_2_stress_status.config(
                        text=f"压力: {'ABORTED' if aborted else 'FAIL'} | {result.completed_episodes}/{result.requested_episodes}",
                        fg=ui.COLOR_GOLD if aborted and result.failures == 0 else ui.COLOR_RED,
                    )
                self.refresh_status()

            self.root.after(0, finish_ui)

        threading.Thread(target=worker, daemon=True).start()

    def cmd_toggle_level_1_2_pause(self):
        ev = getattr(self, "_level_1_2_pause_event", None)
        if ev is None:
            self.log("WARN", "[1-2 LIVE] 当前没有正在运行的实机测试")
            return
        if ev.is_set():
            ev.clear()
            self.log("WARN", "[1-2 LIVE] 已暂停")
        else:
            ev.set()
            self.log("INFO", "[1-2 LIVE] 已继续")

    def cmd_stop_level_1_2_live(self):
        stop_ev = getattr(self, "_level_1_2_stop_event", None)
        pause_ev = getattr(self, "_level_1_2_pause_event", None)
        if stop_ev is None:
            self.log("WARN", "[1-2 LIVE] 当前没有正在运行的实机测试")
            return
        stop_ev.set()
        if pause_ev is not None:
            pause_ev.set()
        self.log("WARN", "[1-2 LIVE] 终止信号已发送")

    def build_with_level_1_2(self, parent):
        original_build_left_panel(self, parent)
        frame = _scroll_frame(parent)
        if frame is None:
            return

        card = self._create_card(frame, "1-2 LIVE 环境测试（会切关）")
        card.pack(fill=tk.X, pady=(0, 10))
        tk.Label(
            card,
            text=(
                "单次 E2E：切到 1-2，处理 intro/选卡，使用独立 1-2 策略完整跑到 WON/LOST。\n"
                "压力测试：连续 5 局，统计环境稳定性与 W/L；不会加入普通“运行全部测试”。"
            ),
            font=("Microsoft YaHei UI", 8),
            fg=ui.CLR_TEXT_MUTED,
            bg=ui.CLR_PANEL,
            justify="left",
            wraplength=700,
        ).pack(anchor="w", padx=12, pady=(0, 8))

        self.lbl_level_1_2_e2e_status = tk.Label(
            card, text="单次: READY", font=ui.FONT_BOLD, fg=ui.COLOR_BLUE, bg=ui.CLR_PANEL
        )
        self.lbl_level_1_2_e2e_status.pack(anchor="w", padx=12, pady=(0, 4))
        self.lbl_level_1_2_stress_status = tk.Label(
            card, text="压力: READY | 5 episodes", font=ui.FONT_BOLD, fg=ui.COLOR_BLUE, bg=ui.CLR_PANEL
        )
        self.lbl_level_1_2_stress_status.pack(anchor="w", padx=12, pady=(0, 8))

        buttons = tk.Frame(card, bg=ui.CLR_PANEL)
        buttons.pack(fill=tk.X, padx=12, pady=(0, 4))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        ui.CyberButton(
            buttons, "[运行单次 1-2 E2E]", self.cmd_run_level_1_2_e2e,
            bg_color=ui.COLOR_GOLD, fg_color="#ffffff", height=36,
        ).grid(row=0, column=0, padx=(0, 4), sticky="ew")
        ui.CyberButton(
            buttons, "[运行 1-2 压力测试 x5]", self.cmd_run_level_1_2_stress,
            bg_color=ui.COLOR_GOLD, fg_color="#ffffff", height=36,
        ).grid(row=0, column=1, padx=(4, 0), sticky="ew")

        ctrl = tk.Frame(card, bg=ui.CLR_PANEL)
        ctrl.pack(fill=tk.X, padx=12, pady=(0, 12))
        ctrl.columnconfigure(0, weight=1)
        ctrl.columnconfigure(1, weight=1)
        ui.CyberButton(
            ctrl, "[暂停 / 继续]", self.cmd_toggle_level_1_2_pause,
            bg_color=ui.CLR_PANEL, fg_color=ui.COLOR_GOLD, height=30,
        ).grid(row=0, column=0, padx=(0, 4), sticky="ew")
        ui.CyberButton(
            ctrl, "[终止]", self.cmd_stop_level_1_2_live,
            bg_color=ui.CLR_PANEL, fg_color=ui.COLOR_RED, height=30,
        ).grid(row=0, column=1, padx=(4, 0), sticky="ew")

    gui_cls.cmd_run_level_1_2_e2e = cmd_run_level_1_2_e2e
    gui_cls.cmd_run_level_1_2_stress = cmd_run_level_1_2_stress
    gui_cls.cmd_toggle_level_1_2_pause = cmd_toggle_level_1_2_pause
    gui_cls.cmd_stop_level_1_2_live = cmd_stop_level_1_2_live
    gui_cls._build_left_panel = build_with_level_1_2
    gui_cls._level_1_2_live_tests_gui_installed = True
