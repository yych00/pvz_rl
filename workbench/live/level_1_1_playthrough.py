# -*- coding: utf-8 -*-
"""Playthrough-aware LIVE contracts and GUI for Adventure 1-1.

This is the 1-1 companion to ``early_levels_live_tests``. It deliberately
reuses the same generic E2E/stress engine and authoritative
``finished_adventure`` playthrough detection so 1-1/1-2/1-3 expose the same
public contract and the same GUI controls.
"""

from __future__ import annotations

import threading
import tkinter as tk
from typing import Any

from .early_levels_live_tests import (
    EarlyLevelSpec,
    EarlyLevelE2EResult,
    EarlyLevelStressResult,
    PLAYTHROUGH_1,
    PLAYTHROUGH_2,
    PLAYTHROUGH_AUTO,
    run_early_level_e2e,
    run_early_level_stress,
)
from .level_1_1_playthrough_policy import (
    LEVEL_1_1_ADVENTURE1_DECK_PRIORITY,
    LEVEL_1_1_ADVENTURE2_DECK_PRIORITY,
    choose_level_1_1_adventure1_plant,
    choose_level_1_1_adventure2_plant,
)


LEVEL_1_1_SPEC = EarlyLevelSpec(
    area=1,
    sub_level=1,
    level_id=1,
    label="1-1",
    adventure1_priority=tuple(LEVEL_1_1_ADVENTURE1_DECK_PRIORITY),
    adventure2_priority=tuple(LEVEL_1_1_ADVENTURE2_DECK_PRIORITY),
    adventure1_policy=choose_level_1_1_adventure1_plant,
    adventure2_policy=choose_level_1_1_adventure2_plant,
    max_plants=40,
)


def run_level_1_1_e2e(client: Any, **kwargs) -> EarlyLevelE2EResult:
    return run_early_level_e2e(client, LEVEL_1_1_SPEC, **kwargs)


def run_level_1_1_adventure1_e2e(client: Any, **kwargs) -> EarlyLevelE2EResult:
    return run_early_level_e2e(
        client, LEVEL_1_1_SPEC, expected_path=PLAYTHROUGH_1, **kwargs
    )


def run_level_1_1_adventure2_e2e(client: Any, **kwargs) -> EarlyLevelE2EResult:
    return run_early_level_e2e(
        client, LEVEL_1_1_SPEC, expected_path=PLAYTHROUGH_2, **kwargs
    )


def run_level_1_1_adventure1_stress(client: Any, **kwargs) -> EarlyLevelStressResult:
    return run_early_level_stress(
        client, LEVEL_1_1_SPEC, expected_path=PLAYTHROUGH_1, **kwargs
    )


def run_level_1_1_adventure2_stress(client: Any, **kwargs) -> EarlyLevelStressResult:
    return run_early_level_stress(
        client, LEVEL_1_1_SPEC, expected_path=PLAYTHROUGH_2, **kwargs
    )


# Compatibility with the old 1-1 API. The generic E2E auto-detects the current
# playthrough; the historical stress alias keeps the same replay-oriented
# convention used by 1-2/1-3 compatibility exports.
def run_level_1_1_stress(client: Any, **kwargs) -> EarlyLevelStressResult:
    return run_level_1_1_adventure2_stress(client, **kwargs)


run_level_1_1_first_run_e2e = run_level_1_1_adventure1_e2e
run_level_1_1_replay_e2e = run_level_1_1_adventure2_e2e


def install_level_1_1_playthrough_gui(gui_cls: Any, ui: Any) -> None:
    """Add the same four playthrough controls used by 1-2 and 1-3."""
    if getattr(gui_cls, "_level_1_1_playthrough_gui_installed", False):
        return

    original_build_left_panel = gui_cls._build_left_panel

    def _scroll_frame(parent):
        canvas = next((w for w in parent.winfo_children() if isinstance(w, tk.Canvas)), None)
        if canvas is None:
            return None
        return next((w for w in canvas.winfo_children() if isinstance(w, tk.Frame)), None)

    def _state(self) -> dict:
        state = getattr(self, "_level_1_1_playthrough_state", None)
        if state is None:
            state = {"running": False, "pause": None, "stop": None}
            self._level_1_1_playthrough_state = state
        return state

    def _start(self, playthrough: str, episodes: int) -> None:
        state = _state(self)
        if state["running"]:
            self.log("WARN", "[1-1 LIVE] 已有实机测试正在运行")
            return

        state["running"] = True
        state["pause"] = threading.Event()
        state["pause"].set()
        state["stop"] = threading.Event()

        mode_text = "一周目" if playthrough == PLAYTHROUGH_1 else "二周目"
        run_text = "E2E" if episodes == 1 else f"x{episodes}"
        self.lbl_level_1_1_playthrough_status.config(
            text=f"{mode_text} {run_text}: RUNNING",
            fg=ui.COLOR_GOLD,
        )
        self.log(
            "HEADER",
            f"\n========== 1-1 {playthrough} {'E2E' if episodes == 1 else f'STRESS x{episodes}'} ==========",
        )

        def progress(phase: str, message: str):
            self.root.after(
                0, lambda: self.log("INFO", f"[1-1:{phase}] {message}")
            )

        def worker():
            if episodes == 1:
                result = run_early_level_e2e(
                    self.client,
                    LEVEL_1_1_SPEC,
                    expected_path=playthrough,
                    progress=progress,
                    pause_event=state["pause"],
                    stop_event=state["stop"],
                )
            else:
                result = run_early_level_stress(
                    self.client,
                    LEVEL_1_1_SPEC,
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

                if episodes == 1:
                    if result.success:
                        self.log("PASS", f"{result.message} | {result.elapsed_ms:.2f}ms")
                        text = f"PASS | {result.actual_path} -> {result.outcome}"
                        color = ui.COLOR_GREEN
                    else:
                        aborted = result.outcome == "ABORTED"
                        self.log(
                            "WARN" if aborted else "FAIL",
                            f"{result.outcome}: {result.message}",
                        )
                        text = f"{'ABORTED' if aborted else 'FAIL'} | {result.outcome}"
                        color = ui.COLOR_GOLD if aborted else ui.COLOR_RED
                else:
                    if result.success:
                        self.log(
                            "PASS",
                            f"1-1 {playthrough} stress PASS: "
                            f"{result.completed_episodes}/{result.requested_episodes}, "
                            f"W/L={result.wins}/{result.losses}, "
                            f"win_rate={result.win_rate:.1%}",
                        )
                        text = f"{mode_text} x{episodes} PASS | W/L {result.wins}/{result.losses}"
                        color = ui.COLOR_GREEN
                    else:
                        aborted = any(
                            r.outcome == "ABORTED" for r in result.episode_results
                        )
                        self.log(
                            "WARN" if aborted and result.failures == 0 else "FAIL",
                            f"1-1 {playthrough} stress "
                            f"{'ABORTED' if aborted else 'FAIL'}: "
                            f"completed={result.completed_episodes}/{result.requested_episodes}, "
                            f"failures={result.failures}, "
                            f"path_mismatch={result.path_mismatches}",
                        )
                        text = f"{mode_text} x{episodes} {'ABORTED' if aborted else 'FAIL'}"
                        color = (
                            ui.COLOR_GOLD
                            if aborted and result.failures == 0
                            else ui.COLOR_RED
                        )

                self.lbl_level_1_1_playthrough_status.config(text=text, fg=color)
                self.refresh_status()

            self.root.after(0, finish_ui)

        threading.Thread(target=worker, daemon=True).start()

    def _toggle_pause(self) -> None:
        state = _state(self)
        ev = state.get("pause")
        if ev is None:
            self.log("WARN", "[1-1 LIVE] 当前没有运行中的测试")
            return
        if ev.is_set():
            ev.clear()
            self.log("WARN", "[1-1 LIVE] 已暂停")
        else:
            ev.set()
            self.log("INFO", "[1-1 LIVE] 已继续")

    def _stop(self) -> None:
        state = _state(self)
        stop_ev = state.get("stop")
        pause_ev = state.get("pause")
        if stop_ev is None:
            self.log("WARN", "[1-1 LIVE] 当前没有运行中的测试")
            return
        stop_ev.set()
        if pause_ev is not None:
            pause_ev.set()
        self.log("WARN", "[1-1 LIVE] 终止信号已发送")

    def build_with_level_1_1_playthrough(self, parent):
        original_build_left_panel(self, parent)
        frame = _scroll_frame(parent)
        if frame is None:
            return

        card = self._create_card(frame, "1-1 LIVE：一周目 / 二周目")
        card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            card,
            text=(
                "周目判定读取 PlayerInfo::mFinishedAdventure：0=一周目，>=1=二周目及以后。\n"
                "一周目使用真实 1-1 教学卡组（豌豆射手）；二周目使用完整 replay 策略。"
            ),
            font=("Microsoft YaHei UI", 8),
            fg=ui.CLR_TEXT_MUTED,
            bg=ui.CLR_PANEL,
            justify="left",
            wraplength=700,
        ).pack(anchor="w", padx=12, pady=(0, 8))

        self.lbl_level_1_1_playthrough_status = tk.Label(
            card,
            text="READY | 一周目 / 二周目",
            font=ui.FONT_BOLD,
            fg=ui.COLOR_BLUE,
            bg=ui.CLR_PANEL,
        )
        self.lbl_level_1_1_playthrough_status.pack(
            anchor="w", padx=12, pady=(0, 8)
        )

        buttons = tk.Frame(card, bg=ui.CLR_PANEL)
        buttons.pack(fill=tk.X, padx=12, pady=(0, 4))
        for c in range(2):
            buttons.columnconfigure(c, weight=1)

        ui.CyberButton(
            buttons,
            "[一周目 E2E]",
            lambda: _start(self, PLAYTHROUGH_1, 1),
            bg_color=ui.COLOR_GOLD,
            fg_color="#ffffff",
            height=36,
        ).grid(row=0, column=0, padx=(0, 4), pady=(0, 4), sticky="ew")
        ui.CyberButton(
            buttons,
            "[二周目 E2E]",
            lambda: _start(self, PLAYTHROUGH_2, 1),
            bg_color=ui.COLOR_GOLD,
            fg_color="#ffffff",
            height=36,
        ).grid(row=0, column=1, padx=(4, 0), pady=(0, 4), sticky="ew")
        ui.CyberButton(
            buttons,
            "[一周目 x5]",
            lambda: _start(self, PLAYTHROUGH_1, 5),
            bg_color=ui.COLOR_GOLD,
            fg_color="#ffffff",
            height=36,
        ).grid(row=1, column=0, padx=(0, 4), sticky="ew")
        ui.CyberButton(
            buttons,
            "[二周目 x5]",
            lambda: _start(self, PLAYTHROUGH_2, 5),
            bg_color=ui.COLOR_GOLD,
            fg_color="#ffffff",
            height=36,
        ).grid(row=1, column=1, padx=(4, 0), sticky="ew")

        controls = tk.Frame(card, bg=ui.CLR_PANEL)
        controls.pack(fill=tk.X, padx=12, pady=(4, 12))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        ui.CyberButton(
            controls,
            "[暂停 / 继续]",
            lambda: _toggle_pause(self),
            bg_color=ui.CLR_PANEL,
            fg_color=ui.COLOR_GOLD,
            height=30,
        ).grid(row=0, column=0, padx=(0, 4), sticky="ew")
        ui.CyberButton(
            controls,
            "[终止]",
            lambda: _stop(self),
            bg_color=ui.CLR_PANEL,
            fg_color=ui.COLOR_RED,
            height=30,
        ).grid(row=0, column=1, padx=(4, 0), sticky="ew")

    gui_cls._build_left_panel = build_with_level_1_1_playthrough
    gui_cls._level_1_1_playthrough_gui_installed = True
