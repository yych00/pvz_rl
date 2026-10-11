# -*- coding: utf-8 -*-
"""Adventure LIVE simulation-speed benchmark.

The benchmark intentionally keeps the four execution models distinct while
using the same logical-tick budget:

1. current stepped mode: repeated ``advance_ticks(chunk)`` calls. This matches
   the current deterministic RL stepping path and therefore pays its sync cost
   for every control chunk.
2. continuous mode: the original game loop remains alive at its current frame
   duration. No pause/injection occurs during the measured interval.
3. animated fast mode: the original game loop remains alive and rendering; only
   the engine frame duration is reduced. There is no per-chunk synchronization
   wait or injected stepping while the measurement runs.
4. persistent headless mode: the original main/render loop is synchronized and
   frozen once, then the same ``chunk`` boundaries are advanced with injected
   Board::Update batches. This preserves agent decision points while removing
   rendering and repeated frame-sync sleeps.

The benchmark advances the currently running battle. It does not reset or roll
back state between modes.
"""

from __future__ import annotations

import ctypes
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SpeedBenchmarkResult:
    key: str
    label: str
    ticks: int
    elapsed_s: float
    ticks_per_second: float
    realtime_multiple: float


def make_speed_result(
    key: str,
    label: str,
    ticks: int,
    elapsed_s: float,
    baseline_ticks_per_second: float,
) -> SpeedBenchmarkResult:
    elapsed = max(float(elapsed_s), 1e-9)
    tps = max(0, int(ticks)) / elapsed
    multiple = tps / baseline_ticks_per_second if baseline_ticks_per_second > 0 else 0.0
    return SpeedBenchmarkResult(
        key=key,
        label=label,
        ticks=max(0, int(ticks)),
        elapsed_s=elapsed,
        ticks_per_second=tps,
        realtime_multiple=multiple,
    )


def install_speed_benchmark_gui(gui_cls: Any, ui: Any) -> None:
    if getattr(gui_cls, "_speed_benchmark_gui_installed", False):
        return

    original_build_live_panel = gui_cls._build_live_panel

    def _bind_speed_native(self) -> bool:
        dll = getattr(getattr(self, "client", None), "dll", None)
        if dll is None:
            return False
        required = (
            "pvz_gym_get_frame_duration",
            "pvz_gym_set_frame_duration",
            "pvz_gym_begin_headless_mode",
            "pvz_gym_advance_ticks_headless",
            "pvz_gym_end_headless_mode",
        )
        if any(not hasattr(dll, name) for name in required):
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

    def _get_frame_duration(self) -> int:
        if not self._bind_speed_native():
            return -1
        return int(self.client.dll.pvz_gym_get_frame_duration())

    def _set_frame_duration(self, value: int) -> bool:
        if not self._bind_speed_native():
            return False
        return bool(self.client.dll.pvz_gym_set_frame_duration(int(value)))

    def _build_speed_benchmark_panel(self, parent):
        self._speed_benchmark_running = False
        self._speed_result_labels = {}
        self.var_speed_ticks = tk.StringVar(value="300")
        self.var_speed_chunk = tk.StringVar(value="30")
        self.var_speed_animated_ms = tk.StringVar(value="1")

        card = tk.Frame(
            parent,
            bg=ui.CLR_PANEL,
            highlightbackground=ui.CLR_BORDER,
            highlightthickness=1,
        )
        card.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
        self._speed_benchmark_panel = card

        header = tk.Frame(card, bg=ui.CLR_PANEL)
        header.pack(fill=tk.X, padx=12, pady=(9, 4))
        tk.Label(
            header,
            text="Simulation Speed Benchmark",
            font=ui.FONT_TITLE,
            fg=ui.CLR_TEXT_MAIN,
            bg=ui.CLR_PANEL,
        ).pack(side=tk.LEFT)
        self.lbl_speed_benchmark_status = tk.Label(
            header,
            text="READY",
            font=ui.FONT_BOLD,
            fg=ui.CLR_TEXT_MUTED,
            bg=ui.CLR_PANEL,
        )
        self.lbl_speed_benchmark_status.pack(side=tk.RIGHT)

        tk.Label(
            card,
            text=(
                "点击下方任一模式名称即可单独测试；测试会真实推进当前游戏状态。"
            ),
            font=("Microsoft YaHei UI", 8),
            fg=ui.CLR_TEXT_MUTED,
            bg=ui.CLR_PANEL,
            anchor="w",
        ).pack(fill=tk.X, padx=12, pady=(0, 7))

        controls = tk.Frame(card, bg=ui.CLR_CARD)
        controls.pack(fill=tk.X, padx=12, pady=(0, 7))

        def field(label: str, variable: tk.StringVar, width: int = 9):
            box = tk.Frame(controls, bg=ui.CLR_CARD)
            box.pack(side=tk.LEFT, padx=(8, 4), pady=7)
            tk.Label(
                box,
                text=label,
                font=("Microsoft YaHei UI", 8, "bold"),
                fg=ui.CLR_TEXT_MUTED,
                bg=ui.CLR_CARD,
            ).pack(anchor="w")
            tk.Entry(
                box,
                textvariable=variable,
                width=width,
                font=ui.FONT_MONO,
                fg=ui.CLR_TEXT_MAIN,
                bg=ui.CLR_PANEL,
                insertbackground=ui.CLR_TEXT_MAIN,
                relief="flat",
                highlightthickness=1,
                highlightbackground=ui.CLR_BORDER,
                highlightcolor=ui.COLOR_BLUE,
            ).pack(pady=(2, 0), ipady=3)

        field("Ticks / mode", self.var_speed_ticks)
        field("Decision chunk", self.var_speed_chunk)
        field("Animated frame ms", self.var_speed_animated_ms)

        # Modes are intentionally started from their individual table rows.
        # Do not expose a sequential four-mode action: each mode advances the
        # real battle and should be selected deliberately.
        self.btn_speed_benchmark = None

        table = tk.Frame(card, bg=ui.CLR_PANEL)
        table.pack(fill=tk.X, padx=12, pady=(0, 9))
        headers = ("Mode", "Ticks", "Elapsed", "Tick/s", "Rate")
        widths = (24, 9, 10, 12, 9)
        for col, (text, width) in enumerate(zip(headers, widths)):
            tk.Label(
                table,
                text=text,
                width=width,
                anchor="w" if col == 0 else "e",
                font=("Microsoft YaHei UI", 8, "bold"),
                fg=ui.CLR_TEXT_MUTED,
                bg=ui.CLR_PANEL,
            ).grid(row=0, column=col, sticky="ew", padx=3)
        table.columnconfigure(0, weight=1)

        rows = (
            ("current", "Current · stepped/sync"),
            ("continuous", "Continuous · no pause"),
            ("animated", "Animated · native loop"),
            ("headless", "Headless · persistent"),
        )
        for row_index, (key, label) in enumerate(rows, start=1):
            values = {}
            for col, (value, width) in enumerate(
                zip((label, "-", "-", "-", "-"), widths)
            ):
                widget = tk.Label(
                    table,
                    text=value,
                    width=width,
                    anchor="w" if col == 0 else "e",
                    font=ui.FONT_MONO if col else ("Microsoft YaHei UI", 8),
                    fg=ui.CLR_TEXT_MAIN if col == 0 else ui.COLOR_BLUE,
                    bg=ui.CLR_PANEL,
                )
                widget.grid(row=row_index, column=col, sticky="ew", padx=3, pady=1)
                if col > 0:
                    values[("ticks", "elapsed", "tps", "rate")[col - 1]] = widget
            self._speed_result_labels[key] = values

        self.lbl_speed_baseline = tk.Label(
            card,
            text="Baseline: -",
            font=("Microsoft YaHei UI", 8),
            fg=ui.CLR_TEXT_MUTED,
            bg=ui.CLR_PANEL,
            anchor="w",
        )
        self.lbl_speed_baseline.pack(fill=tk.X, padx=12, pady=(0, 8))

    def _build_live_panel_with_speed(self, parent):
        # Build the scene tests first. The original LIVE builder owns the one
        # scrollable canvas used by this tab and exposes its content frame as
        # ``_live_scrollable_frame``.
        original_build_live_panel(self, parent)

        # Keep the speed benchmark in that same content frame, after the scene
        # tests. This gives the whole LIVE surface one continuous scrollbar
        # instead of a fixed benchmark above a separately scrolling test list.
        live_content = getattr(self, "_live_scrollable_frame", None)
        if live_content is not None:
            self._build_speed_benchmark_panel(live_content)

    def _reset_speed_rows(self):
        for labels in self._speed_result_labels.values():
            for widget in labels.values():
                widget.config(text="-")

    def _show_speed_result(self, result: SpeedBenchmarkResult):
        labels = self._speed_result_labels[result.key]
        labels["ticks"].config(text=str(result.ticks))
        labels["elapsed"].config(text=f"{result.elapsed_s:.3f}s")
        labels["tps"].config(text=f"{result.ticks_per_second:,.1f}")
        labels["rate"].config(text=f"{result.realtime_multiple:.2f}x")
        self.log(
            "PASS",
            f"[SPEED] {result.label}: {result.ticks} ticks / {result.elapsed_s:.4f}s "
            f"= {result.ticks_per_second:.1f} tick/s = {result.realtime_multiple:.2f}x",
        )

    def _start_speed_benchmark(self):
        if self._speed_benchmark_running:
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
            total_ticks = int(self.var_speed_ticks.get())
            chunk = int(self.var_speed_chunk.get())
            animated_ms = int(self.var_speed_animated_ms.get())
            if total_ticks <= 0 or chunk <= 0 or animated_ms <= 0:
                raise ValueError("all benchmark values must be positive")
            if chunk > total_ticks:
                raise ValueError("Decision chunk must not exceed Ticks / mode")
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

        self._speed_benchmark_running = True
        self._live_running = True
        self._static_test_running = True
        button = getattr(self, "btn_speed_benchmark", None)
        if button is not None:
            button.set_disabled(True)
        self.lbl_speed_benchmark_status.config(text="RUNNING", fg=ui.COLOR_BLUE)
        self._reset_speed_rows()
        threading.Thread(
            target=self._speed_benchmark_worker,
            args=(total_ticks, chunk, animated_ms),
            daemon=True,
        ).start()

    def _speed_benchmark_worker(self, total_ticks: int, chunk: int, animated_ms: int):
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

            # Mode 1: current deterministic training shape. Every control chunk
            # enters advance_ticks(), which performs its own main-loop freeze and
            # synchronization sleep.
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
                phase, _ = self.client.get_ui()
                if phase != 3:
                    break
            elapsed = time.perf_counter() - started
            current = make_speed_result(
                "current", "Current stepped/sync", advanced_total, elapsed, baseline_tps
            )
            self.root.after(0, self._show_speed_result, current)
            if advanced_total < total_ticks:
                raise RuntimeError(
                    f"battle ended during current mode ({advanced_total}/{total_ticks} ticks)"
                )

            # Mode 2: untouched original main loop. This is the Adventure LIVE
            # continuous/no-pause gear: no injected updates and no frame override.
            phase, _ = self.client.get_ui()
            if phase != 3:
                raise RuntimeError("battle is no longer PLAYING before continuous mode")
            continuous_start_clock = int(self.client.get_game_clock())
            continuous_last_clock = continuous_start_clock
            started = time.perf_counter()
            if not self.client.resume():
                raise RuntimeError("could not resume battle for continuous mode")
            deadline = started + max(3.0, total_ticks * original_ms / 1000.0 * 20.0 + 2.0)
            while time.perf_counter() < deadline:
                phase, _ = self.client.get_ui()
                if phase != 3:
                    break
                observed = int(self.client.get_game_clock())
                if observed >= continuous_start_clock:
                    continuous_last_clock = max(continuous_last_clock, observed)
                if continuous_last_clock - continuous_start_clock >= total_ticks:
                    break
                time.sleep(0)
            if self.client.get_ui()[0] == 3:
                self.client.pause()
            elapsed = time.perf_counter() - started
            continuous_ticks = max(0, continuous_last_clock - continuous_start_clock)
            continuous = make_speed_result(
                "continuous",
                "Continuous no pause",
                continuous_ticks,
                elapsed,
                baseline_tps,
            )
            self.root.after(0, self._show_speed_result, continuous)
            if continuous_ticks < total_ticks:
                raise RuntimeError(
                    f"continuous mode stopped early ({continuous_ticks}/{total_ticks} ticks)"
                )

            # Mode 3: original main loop + rendering at a shorter frame duration.
            # There is no injected step and no game-frame synchronization delay.
            phase, _ = self.client.get_ui()
            if phase != 3:
                raise RuntimeError("battle is no longer PLAYING before animated mode")
            if not self._set_frame_duration(animated_ms):
                raise RuntimeError(f"failed to set animated frame duration to {animated_ms} ms")
            start_clock = int(self.client.get_game_clock())
            last_clock = start_clock
            started = time.perf_counter()
            if not self.client.resume():
                raise RuntimeError("could not resume battle for animated mode")
            deadline = started + max(3.0, total_ticks * animated_ms / 1000.0 * 20.0 + 2.0)
            while time.perf_counter() < deadline:
                phase, _ = self.client.get_ui()
                if phase != 3:
                    break
                observed = int(self.client.get_game_clock())
                if observed >= start_clock:
                    last_clock = max(last_clock, observed)
                if last_clock - start_clock >= total_ticks:
                    break
                # Yield only to the Python scheduler. The PvZ main loop itself
                # remains free-running; there is no per-frame/chunk wait here.
                time.sleep(0)
            if self.client.get_ui()[0] == 3:
                self.client.pause()
            elapsed = time.perf_counter() - started
            animated_ticks = max(0, last_clock - start_clock)
            if not self._set_frame_duration(original_ms):
                raise RuntimeError("failed to restore original frame duration")
            animated = make_speed_result(
                "animated", "Animated native loop", animated_ticks, elapsed, baseline_tps
            )
            self.root.after(0, self._show_speed_result, animated)
            if animated_ticks < total_ticks:
                raise RuntimeError(
                    f"animated mode stopped early ({animated_ticks}/{total_ticks} ticks)"
                )

            # Mode 4: same decision chunk as mode 1, but freeze/render-sync only
            # once. Subsequent chunks call Board::Update directly while the
            # original main/render loop remains blocked.
            phase, _ = self.client.get_ui()
            if phase != 3:
                raise RuntimeError("battle is no longer PLAYING before headless mode")
            started = time.perf_counter()
            headless_active = bool(self.client.dll.pvz_gym_begin_headless_mode())
            if not headless_active:
                raise RuntimeError("could not enter persistent headless mode")
            headless_total = 0
            while headless_total < total_ticks:
                request = min(chunk, total_ticks - headless_total)
                advanced = int(self.client.dll.pvz_gym_advance_ticks_headless(request))
                if advanced <= 0:
                    break
                headless_total += advanced
                phase, _ = self.client.get_ui()
                if phase != 3:
                    break
            if not bool(self.client.dll.pvz_gym_end_headless_mode()):
                raise RuntimeError("failed to leave persistent headless mode")
            headless_active = False
            elapsed = time.perf_counter() - started
            headless = make_speed_result(
                "headless", "Headless persistent", headless_total, elapsed, baseline_tps
            )
            self.root.after(0, self._show_speed_result, headless)
            if headless_total < total_ticks:
                raise RuntimeError(
                    f"headless mode stopped early ({headless_total}/{total_ticks} ticks)"
                )
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
            self.root.after(0, self._finish_speed_benchmark, error)

    def _finish_speed_benchmark(self, error: Exception | None):
        self._speed_benchmark_running = False
        self._live_running = False
        self._static_test_running = False
        button = getattr(self, "btn_speed_benchmark", None)
        if button is not None:
            button.set_disabled(False)
        if error is None:
            self.lbl_speed_benchmark_status.config(text="DONE", fg=ui.COLOR_GREEN)
            self.log("PASS", "[SPEED] four-mode benchmark complete")
        else:
            self.lbl_speed_benchmark_status.config(text="STOPPED", fg=ui.COLOR_RED)
            self.log("FAIL", f"[SPEED] {error}")
        self.refresh_status()

    gui_cls._bind_speed_native = _bind_speed_native
    gui_cls._get_frame_duration = _get_frame_duration
    gui_cls._set_frame_duration = _set_frame_duration
    gui_cls._build_speed_benchmark_panel = _build_speed_benchmark_panel
    gui_cls._reset_speed_rows = _reset_speed_rows
    gui_cls._show_speed_result = _show_speed_result
    gui_cls._start_speed_benchmark = _start_speed_benchmark
    gui_cls._speed_benchmark_worker = _speed_benchmark_worker
    gui_cls._finish_speed_benchmark = _finish_speed_benchmark
    gui_cls._build_live_panel = _build_live_panel_with_speed
    gui_cls._speed_benchmark_gui_installed = True


__all__ = [
    "SpeedBenchmarkResult",
    "make_speed_result",
    "install_speed_benchmark_gui",
]
