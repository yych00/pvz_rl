# -*- coding: utf-8 -*-
"""Compact LIVE cards for isolated early-level replay tests.

Each level exposes only two buttons: E2E and E2E x5. The current single-level
playthrough mode is read from native runtime memory when a run starts, then
locked for the whole run. Stress therefore repeats five episodes in the same
mode even if the just-finished episode mutates profile progress.
"""

from __future__ import annotations

import threading
import tkinter as tk
from typing import Any

from . import early_levels_live_tests as _early
from .early_levels_live_tests import LEVEL_1_2_SPEC, LEVEL_1_3_SPEC
from .level_1_1_playthrough_live_tests import LEVEL_1_1_SPEC


def _detect_playthrough(client: Any) -> tuple[str, int]:
    state = client.get_runtime_state()
    if not isinstance(state, dict) or state.get("error"):
        raise RuntimeError(
            str(state.get("error")) if isinstance(state, dict) else "runtime state unavailable"
        )
    try:
        finished = int(state.get("finished_adventure", -1))
    except Exception as exc:
        raise RuntimeError("finished_adventure is not readable") from exc
    if finished < 0:
        raise RuntimeError("finished_adventure is unavailable; rebuild pvz_gym_native.dll")
    return (
        _early.PLAYTHROUGH_1 if finished == 0 else _early.PLAYTHROUGH_2,
        finished,
    )


def install_compact_live_tests_gui(gui_cls: Any, ui: Any) -> None:
    if getattr(gui_cls, "_compact_live_tests_gui_installed", False):
        return

    original_build_left_panel = gui_cls._build_left_panel
    specs = (LEVEL_1_1_SPEC, LEVEL_1_2_SPEC, LEVEL_1_3_SPEC)

    def _scroll_frame(parent):
        canvas = next((w for w in parent.winfo_children() if isinstance(w, tk.Canvas)), None)
        if canvas is None:
            return None
        return next((w for w in canvas.winfo_children() if isinstance(w, tk.Frame)), None)

    def _states(self) -> dict:
        states = getattr(self, "_compact_live_states", None)
        if states is None:
            states = {}
            self._compact_live_states = states
        return states

    def _state(self, spec) -> dict:
        return _states(self).setdefault(spec.label, {"running": False})

    def _status(self, spec):
        return getattr(self, f"lbl_compact_live_{spec.level_id}_status", None)

    def _start(self, spec, episodes: int) -> None:
        state = _state(self, spec)
        if state["running"]:
            self.log("WARN", f"[{spec.label} LIVE] 测试正在运行")
            return

        state["running"] = True
        label = _status(self, spec)
        if label is not None:
            label.config(text="RUNNING", fg=ui.COLOR_GOLD)

        run_name = "E2E" if episodes == 1 else "E2E x5"
        self.log("HEADER", f"\n========== {spec.label} {run_name} ==========")

        def progress(phase: str, message: str) -> None:
            self.root.after(
                0,
                lambda p=phase, m=message: self.log(
                    "INFO", f"[{spec.label}:{p}] {m}"
                ),
            )

        def worker() -> None:
            try:
                playthrough, finished = _detect_playthrough(self.client)
                self.root.after(
                    0,
                    lambda: self.log(
                        "INFO",
                        f"[{spec.label}] memory mode detected: finished_adventure={finished}",
                    ),
                )

                # The explicit path is intentionally captured once here. The
                # per-episode patch re-applies that same native mode before
                # every episode, so x5 cannot drift after episode 1 completes.
                if episodes == 1:
                    result = _early.run_early_level_e2e(
                        self.client,
                        spec,
                        expected_path=playthrough,
                        progress=progress,
                    )
                else:
                    result = _early.run_early_level_stress(
                        self.client,
                        spec,
                        episodes=episodes,
                        expected_path=playthrough,
                        progress=progress,
                    )
            except Exception as exc:
                result = exc

            def finish_ui() -> None:
                state["running"] = False
                status = _status(self, spec)

                if isinstance(result, Exception):
                    self.log("FAIL", f"{spec.label} {run_name}: {result}")
                    if status is not None:
                        status.config(text="FAIL", fg=ui.COLOR_RED)
                    self.refresh_status()
                    return

                if episodes == 1:
                    if result.success:
                        self.log("PASS", f"{result.message} | {result.elapsed_ms:.2f}ms")
                        text = f"PASS · {result.outcome}"
                        color = ui.COLOR_GREEN
                    else:
                        self.log("FAIL", f"{result.outcome}: {result.message}")
                        text = f"FAIL · {result.outcome}"
                        color = ui.COLOR_RED
                else:
                    if result.success:
                        self.log(
                            "PASS",
                            f"{spec.label} x5 PASS: "
                            f"{result.completed_episodes}/{result.requested_episodes}, "
                            f"W/L={result.wins}/{result.losses}, "
                            f"win_rate={result.win_rate:.1%}",
                        )
                        text = f"PASS · {result.wins}W/{result.losses}L"
                        color = ui.COLOR_GREEN
                    else:
                        self.log(
                            "FAIL",
                            f"{spec.label} x5 FAIL: "
                            f"completed={result.completed_episodes}/{result.requested_episodes}, "
                            f"failures={result.failures}",
                        )
                        text = "FAIL"
                        color = ui.COLOR_RED

                if status is not None:
                    status.config(text=text, fg=color)
                self.refresh_status()

            self.root.after(0, finish_ui)

        threading.Thread(target=worker, daemon=True).start()

    def _build_card(self, frame, spec) -> None:
        card = self._create_card(frame, f"{spec.label} LIVE")
        card.pack(fill=tk.X, pady=(0, 8))

        status = tk.Label(
            card,
            text="READY",
            font=ui.FONT_BOLD,
            fg=ui.COLOR_BLUE,
            bg=ui.CLR_PANEL,
        )
        setattr(self, f"lbl_compact_live_{spec.level_id}_status", status)
        status.pack(anchor="w", padx=12, pady=(0, 8))

        buttons = tk.Frame(card, bg=ui.CLR_PANEL)
        buttons.pack(fill=tk.X, padx=12, pady=(0, 12))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)

        ui.CyberButton(
            buttons,
            "E2E",
            lambda sp=spec: _start(self, sp, 1),
            bg_color=ui.COLOR_GOLD,
            fg_color="#ffffff",
            height=38,
        ).grid(row=0, column=0, padx=(0, 5), sticky="ew")

        ui.CyberButton(
            buttons,
            "E2E x5",
            lambda sp=spec: _start(self, sp, 5),
            bg_color=ui.CLR_PANEL,
            fg_color=ui.COLOR_GOLD,
            height=38,
        ).grid(row=0, column=1, padx=(5, 0), sticky="ew")

    def build_with_compact_live(self, parent):
        original_build_left_panel(self, parent)
        frame = _scroll_frame(parent)
        if frame is None:
            return
        for spec in specs:
            _build_card(self, frame, spec)

    gui_cls._build_left_panel = build_with_compact_live
    gui_cls._compact_live_tests_gui_installed = True
