# -*- coding: utf-8 -*-
"""Clickable per-row runners for the simulation speed benchmark."""

from __future__ import annotations

import threading
import time
from typing import Any


_MODE_LABELS = {
    "Current · stepped/sync": "current",
    "Continuous · no pause": "continuous",
    "Animated · native loop": "animated",
    "Headless · persistent": "headless",
}


def install_speed_benchmark_clicks(gui_cls: Any, ui: Any) -> None:
    if getattr(gui_cls, "_speed_benchmark_clicks_installed", False):
        return

    original_build_panel = gui_cls._build_speed_benchmark_panel

    def _walk(widget):
        yield widget
        for child in widget.winfo_children():
            yield from _walk(child)

    def _build_speed_benchmark_panel_with_clicks(self, parent):
        original_build_panel(self, parent)
        panel = getattr(self, "_speed_benchmark_panel", None)
        if panel is None:
            return
        self._speed_mode_name_widgets = {}
        for widget in _walk(panel):
            try:
                text = str(widget.cget("text"))
            except Exception:
                continue
            mode = _MODE_LABELS.get(text)
            if mode is None:
                continue
            try:
                widget.configure(fg=ui.COLOR_BLUE, cursor="hand2")
                widget.bind(
                    "<Button-1>",
                    lambda _event, key=mode: self._start_speed_benchmark_mode(key),
                )
                widget.bind(
                    "<Enter>",
                    lambda _event, w=widget: w.configure(fg=ui.CLR_TEXT_MAIN),
                )
                widget.bind(
                    "<Leave>",
                    lambda _event, w=widget: w.configure(fg=ui.COLOR_BLUE),
                )
            except Exception:
                pass
            self._speed_mode_name_widgets[mode] = widget

    def _parse_speed_config(self):
        total_ticks = int(self.var_speed_ticks.get())
        chunk = int(self.var_speed_chunk.get())
        animated_ms = int(self.var_speed_animated_ms.get())
        if total_ticks <= 0 or chunk <= 0 or animated_ms <= 0:
            raise ValueError("all benchmark values must be positive")
        if chunk > total_ticks:
            raise ValueError("Decision chunk must not exceed Ticks / mode")
        return total_ticks, chunk, animated_ms

    def _start_speed_benchmark_mode(self, mode: str):
        mode = str(mode).strip().lower()
        if mode not in {"current", "continuous", "animated", "headless"}:
            self.log("FAIL", f"[SPEED] unknown mode: {mode}")
            return
        if getattr(self, "_speed_benchmark_running", False):
            return
        if getattr(self, "_live_running", False):
            self.log("WARN", "[SPEED] Adventure LIVE task is already running")
            return
        if getattr(self, "_training_running", False):
            self.log("WARN", "[SPEED] Gym Training is running")
            return
        if getattr(self, "_static_test_running", False):
            self.log("WARN", "[SPEED] Native test is running")
            return

        try:
            total_ticks, chunk, animated_ms = self._parse_speed_config()
        except Exception as exc:
            self.log("FAIL", f"[SPEED] config error: {exc}")
            return

        if not self._bind_speed_native():
            self.log(
                "FAIL",
                "[SPEED] current DLL lacks speed/headless exports; rebuild native DLL from this branch",
            )
            self.lbl_speed_benchmark_status.config(text="REBUILD DLL", fg=ui.COLOR_RED)
            return

        ok, message = self.client.attach()
        if not ok:
            self.log("FAIL", f"[SPEED] attach failed: {message}")
            return
        phase, _ = self.client.get_ui()
        if phase != 3:
            self.log("WARN", "[SPEED] enter an Adventure battle first (UI must be PLAYING)")
            return

        labels = self._speed_result_labels.get(mode, {})
        for widget in labels.values():
            widget.config(text="-")

        self._speed_benchmark_running = True
        self._live_running = True
        self._static_test_running = True
        self._speed_single_mode = mode
        self.lbl_speed_benchmark_status.config(text=mode.upper(), fg=ui.COLOR_BLUE)
        threading.Thread(
            target=self._speed_benchmark_mode_worker,
            args=(mode, total_ticks, chunk, animated_ms),
            daemon=True,
        ).start()

    def _wait_native_ticks(self, total_ticks: int, frame_ms: int, *, start_clock: int):
        last_clock = int(start_clock)
        started = time.perf_counter()
        deadline = started + max(3.0, total_ticks * frame_ms / 1000.0 * 20.0 + 2.0)
        while time.perf_counter() < deadline:
            phase, _ = self.client.get_ui()
            if phase != 3:
                break
            observed = int(self.client.get_game_clock())
            if observed >= start_clock:
                last_clock = max(last_clock, observed)
            if last_clock - start_clock >= total_ticks:
                break
            time.sleep(0)
        return max(0, last_clock - start_clock), time.perf_counter() - started

    def _speed_benchmark_mode_worker(
        self,
        mode: str,
        total_ticks: int,
        chunk: int,
        animated_ms: int,
    ):
        original_ms = -1
        original_paused = False
        headless_active = False
        error: Exception | None = None
        try:
            original_ms = self._get_frame_duration()
            if original_ms <= 0:
                raise RuntimeError("could not read engine frame duration")
            baseline_tps = 1000.0 / float(original_ms)
            original_paused = bool(self.client.is_paused())
            baseline_text = (
                f"Baseline: frame {original_ms} ms = {baseline_tps:.1f} tick/s = 1.00x · "
                f"animated target {animated_ms} ms · chunk {chunk}"
            )
            self.root.after(0, lambda: self.lbl_speed_baseline.config(text=baseline_text))

            if mode == "current":
                if not self.client.pause() and not self.client.is_paused():
                    raise RuntimeError("could not pause battle for current stepped mode")
                advanced_total = 0
                started = time.perf_counter()
                while advanced_total < total_ticks:
                    request = min(chunk, total_ticks - advanced_total)
                    advanced = int(self.client.advance_ticks(request))
                    if advanced <= 0:
                        break
                    advanced_total += advanced
                    if self.client.get_ui()[0] != 3:
                        break
                elapsed = time.perf_counter() - started
                result = self._make_single_speed_result(
                    "current", "Current stepped/sync", advanced_total, elapsed, baseline_tps
                )
                if advanced_total < total_ticks:
                    raise RuntimeError(
                        f"battle ended during current mode ({advanced_total}/{total_ticks} ticks)"
                    )

            elif mode == "continuous":
                if not self.client.resume():
                    raise RuntimeError("could not resume battle for continuous mode")
                start_clock = int(self.client.get_game_clock())
                ticks, elapsed = self._wait_native_ticks(
                    total_ticks, original_ms, start_clock=start_clock
                )
                if self.client.get_ui()[0] == 3:
                    self.client.pause()
                result = self._make_single_speed_result(
                    "continuous", "Continuous no pause", ticks, elapsed, baseline_tps
                )
                if ticks < total_ticks:
                    raise RuntimeError(
                        f"continuous mode stopped early ({ticks}/{total_ticks} ticks)"
                    )

            elif mode == "animated":
                if not self._set_frame_duration(animated_ms):
                    raise RuntimeError(
                        f"failed to set animated frame duration to {animated_ms} ms"
                    )
                if not self.client.resume():
                    raise RuntimeError("could not resume battle for animated mode")
                start_clock = int(self.client.get_game_clock())
                ticks, elapsed = self._wait_native_ticks(
                    total_ticks, animated_ms, start_clock=start_clock
                )
                if self.client.get_ui()[0] == 3:
                    self.client.pause()
                if not self._set_frame_duration(original_ms):
                    raise RuntimeError("failed to restore original frame duration")
                result = self._make_single_speed_result(
                    "animated", "Animated native loop", ticks, elapsed, baseline_tps
                )
                if ticks < total_ticks:
                    raise RuntimeError(
                        f"animated mode stopped early ({ticks}/{total_ticks} ticks)"
                    )

            else:
                started = time.perf_counter()
                headless_active = bool(self.client.dll.pvz_gym_begin_headless_mode())
                if not headless_active:
                    raise RuntimeError("could not enter headless mode")

                headless_total = 0
                terminal_ui: int | None = None
                while headless_total < total_ticks:
                    request = min(chunk, total_ticks - headless_total)
                    advanced = int(self.client.dll.pvz_gym_advance_ticks_headless(request))
                    phase, _ = self.client.get_ui()

                    if advanced < 0:
                        raise RuntimeError(
                            f"headless advance failed: code={advanced}, ui={phase}, "
                            f"advanced={headless_total}/{total_ticks} ticks"
                        )
                    if advanced == 0:
                        if phase != 3:
                            terminal_ui = int(phase)
                            break
                        raise RuntimeError(
                            f"headless made no progress while still PLAYING: "
                            f"ui={phase}, advanced={headless_total}/{total_ticks} ticks"
                        )

                    headless_total += advanced
                    if phase != 3:
                        terminal_ui = int(phase)
                        break

                if not bool(self.client.dll.pvz_gym_end_headless_mode()):
                    raise RuntimeError("failed to leave headless mode")
                headless_active = False
                elapsed = time.perf_counter() - started
                result = self._make_single_speed_result(
                    "headless", "Headless persistent", headless_total, elapsed, baseline_tps
                )

                if terminal_ui is not None:
                    self.root.after(
                        0,
                        self.log,
                        "WARN",
                        f"[SPEED] headless battle left PLAYING normally: "
                        f"ui={terminal_ui}, advanced={headless_total}/{total_ticks} ticks",
                    )
                elif headless_total < total_ticks:
                    raise RuntimeError(
                        f"headless stopped unexpectedly while PLAYING "
                        f"({headless_total}/{total_ticks} ticks)"
                    )

            self.root.after(0, self._show_speed_result, result)
        except Exception as exc:
            error = exc
        finally:
            if headless_active:
                try:
                    self.client.dll.pvz_gym_end_headless_mode()
                except Exception:
                    pass
            if original_ms > 0:
                try:
                    self._set_frame_duration(original_ms)
                except Exception:
                    pass
            try:
                phase, _ = self.client.get_ui()
                if phase == 3:
                    if original_paused:
                        self.client.pause()
                    else:
                        self.client.resume()
            except Exception:
                pass
            self.root.after(0, self._finish_speed_benchmark_mode, mode, error)

    def _make_single_speed_result(
        self,
        key: str,
        label: str,
        ticks: int,
        elapsed_s: float,
        baseline_tps: float,
    ):
        # Import lazily to avoid changing the original speed-benchmark module's
        # install order or creating an import cycle.
        from .speed_benchmark import make_speed_result

        return make_speed_result(key, label, ticks, elapsed_s, baseline_tps)

    def _finish_speed_benchmark_mode(self, mode: str, error: Exception | None):
        self._speed_benchmark_running = False
        self._live_running = False
        self._static_test_running = False
        self._speed_single_mode = None
        if error is None:
            self.lbl_speed_benchmark_status.config(text="DONE", fg=ui.COLOR_GREEN)
            self.log("PASS", f"[SPEED] {mode} benchmark complete")
        else:
            self.lbl_speed_benchmark_status.config(text="STOPPED", fg=ui.COLOR_RED)
            self.log("FAIL", f"[SPEED] {mode}: {error}")
        self.refresh_status()

    gui_cls._build_speed_benchmark_panel = _build_speed_benchmark_panel_with_clicks
    gui_cls._parse_speed_config = _parse_speed_config
    gui_cls._start_speed_benchmark_mode = _start_speed_benchmark_mode
    gui_cls._wait_native_ticks = _wait_native_ticks
    gui_cls._speed_benchmark_mode_worker = _speed_benchmark_mode_worker
    gui_cls._make_single_speed_result = _make_single_speed_result
    gui_cls._finish_speed_benchmark_mode = _finish_speed_benchmark_mode
    gui_cls._speed_benchmark_clicks_installed = True


__all__ = ["install_speed_benchmark_clicks"]
