# -*- coding: utf-8 -*-
"""Unified Gym Training tab for integration baselines and RL trainers."""

from __future__ import annotations

import json
import threading
from dataclasses import replace
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any

from pvz_gym.client import PvZClient
from pvz_gym.env import PvZEnv
from pvz_gym.random_runner import RandomAgentConfig, RandomAgentSummary, run_random_agent
from train.common import EnvConfig
from train.dqn.agent import DoubleDQNAgent
from train.dqn.config import DoubleDQNConfig, validate_resume_config
from train.dqn.controller import DoubleDQNSummary, DoubleDQNTrainerController
from train.dqn.evaluate import EvaluationSummary, evaluate_checkpoint
from train.dqn.exploration import (
    DEFAULT_EXPLORATION_RATIO, EXPLORATION_MODE_LABELS, format_exploration_ratio, parse_action_family_weights,
    parse_exploration_ratio, q_guided_fraction,
)
from train.dqn.run_state import load_checkpoint_seed_types, load_episode_summary, load_run_seed_types
from train.registry import all_trainer_specs, get_trainer_spec, trainer_key_from_label
from .dqn_headless import _selected_mode
from .training_runtime import runtime_settings
from .deck_panel import DeckPanel, start_deck_test


GREEDY_TREND_COLOR = "#f59e0b"


def _format_dqn_mower_log(data: dict) -> str:
    """Keep the actual mower penalty distinct from the net transition reward."""
    parts = [f"[TRAIN:DQN] mower used x{data['mowers_used']}"]
    if data.get("reward_scheme") is not None:
        parts.append(f"scheme={data['reward_scheme']}")
    breakdown = data.get("reward_breakdown") or {}
    mower_reward = breakdown.get("mower_reward", data.get("mower_reward"))
    if mower_reward is not None:
        parts.append(f"mower_penalty={float(mower_reward):+.3f}")
    if "shaping_reward" in breakdown:
        parts.append("first_loss=yes" if breakdown["first_mower_reward"] < 0 else "first_loss=no")
        parts.append(f"shaping={float(breakdown['shaping_reward']):+.3f}")
        parts.append(f"terminal={float(breakdown['terminal_reward']):+.3f}")
    elif "stage_reward" in breakdown:
        parts.append(f"stage_reward={float(breakdown['stage_reward']):+.3f}")
        parts.append(f"terminal_reward={float(breakdown['terminal_reward']):+.3f}")
    elif mower_reward is not None:
        other_reward = float(data["training_reward"]) - float(mower_reward)
        parts.append(f"other_reward={other_reward:+.3f}")
    parts.append(f"step_reward={float(data['training_reward']):+.3f}")
    return " · ".join(parts)


def _new_training_history() -> dict[str, list]:
    return {
        "episode": [], "outcome": [], "recent_win_rate": [],
        "win_rate": [], "reward": [], "recent_reward": [],
        "mean_reward": [], "loss": [], "recent_loss": [],
        "mean_loss": [], "loss_samples": [], "evaluation": [],
    }


def _source_trend_records(gui: Any) -> list[dict]:
    """Records carried into training; checkpoint previews show history in any mode."""
    mode_var = getattr(gui, "var_training_checkpoint_load_mode", None)
    if mode_var is not None:
        from .checkpoint_load_mode import _is_resume_label
        if not _is_resume_label(mode_var.get()):
            return []
    return (
        gui._training_checkpoint_records
        if getattr(gui, "_training_include_source_history", True)
        else []
    )


def _load_trend_records(path: Path, up_to_episode: int) -> list[dict]:
    """Read the selected checkpoint's recorded episodes in episode order."""
    if not path.is_file() or up_to_episode <= 0:
        return []
    records: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            episode = int(record.get("episode", 0) or 0)
            if 0 < episode <= up_to_episode:
                records[episode] = record
    return [records[episode] for episode in sorted(records)]


def _source_evaluation_records(gui: Any) -> list[dict]:
    mode_var = getattr(gui, "var_training_checkpoint_load_mode", None)
    if mode_var is not None:
        from .checkpoint_load_mode import _is_resume_label
        if not _is_resume_label(mode_var.get()):
            return []
    return (getattr(gui, "_training_checkpoint_evaluations", [])
            if getattr(gui, "_training_include_source_history", True) else [])


def _seed_history_file(path: Path, records: list[dict]) -> None:
    """Carry history into a new run without replacing an existing run's log."""
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError:
        return
    with handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _load_evaluation_records(path: Path, up_to_episode: int, up_to_step: int) -> list[dict]:
    """Exclude partial evaluations and results newer than the selected checkpoint."""
    if not path.is_file():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                if (isinstance(record, dict) and record.get("complete")
                        and 0 < int(record.get("episodes", 0))
                        and 0 <= int(record["episode"]) <= up_to_episode
                        and 0 <= int(record["env_step"]) <= up_to_step
                        and 0.0 <= float(record["win_rate"]) <= 1.0):
                    records.append(record)
            except (ValueError, TypeError, KeyError):
                continue
    return sorted(records, key=lambda record: (int(record["episode"]), int(record["env_step"])))


def _trend_segments(values: list[float | None], columns: int) -> list[list[int]]:
    """Keep the full history in memory, but bound canvas points by pixel columns."""
    segments: list[list[int]] = []
    segment: list[int] = []
    bucket: list[int] = []
    bucket_id = -1

    def flush_bucket() -> None:
        if bucket:
            low = min(bucket, key=lambda index: values[index])
            high = max(bucket, key=lambda index: values[index])
            segment.extend(sorted({bucket[0], low, high, bucket[-1]}))
            bucket.clear()

    for index, value in enumerate(values):
        if value is None:
            flush_bucket()
            if segment:
                segments.append(segment)
                segment = []
            bucket_id = -1
            continue
        column = index * max(columns, 1) // max(len(values), 1)
        if bucket and column != bucket_id:
            flush_bucket()
        bucket_id = column
        bucket.append(index)
    flush_bucket()
    if segment:
        segments.append(segment)
    return segments


def install_training_gui(gui_cls: Any, ui: Any) -> None:
    if getattr(gui_cls, "_training_gui_installed", False):
        return

    original_build_left_panel = gui_cls._build_left_panel
    original_begin_live = gui_cls._begin_live

    def _card(parent, title: str):
        frame = tk.Frame(
            parent,
            bg=ui.CLR_CARD,
            highlightbackground=ui.CLR_BORDER,
            highlightthickness=1,
        )
        frame.pack(fill=tk.X, padx=12, pady=(0, 8))
        tk.Label(
            frame,
            text=title,
            font=ui.FONT_BOLD,
            fg=ui.COLOR_BLUE,
            bg=ui.CLR_CARD,
        ).pack(anchor="w", padx=10, pady=(8, 5))
        body = tk.Frame(frame, bg=ui.CLR_CARD)
        body.pack(fill=tk.X, padx=10, pady=(0, 9))
        return frame, body

    def _entry_field(parent, label: str, variable: tk.StringVar, width: int = 12):
        box = tk.Frame(parent, bg=parent["bg"])
        tk.Label(
            box,
            text=label,
            font=("Microsoft YaHei UI", 8, "bold"),
            fg=ui.CLR_TEXT_MUTED,
            bg=parent["bg"],
        ).pack(anchor="w")
        entry = tk.Entry(
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
            disabledbackground=ui.CLR_PANEL,
            disabledforeground=ui.CLR_TEXT_MUTED,
        )
        entry.pack(fill=tk.X, pady=(3, 0), ipady=4)
        return box, entry

    def _parameter_field(parent, index, label, variable, choices=None):
        column = index % 3
        box = tk.Frame(
            parent, bg=ui.CLR_PANEL,
            highlightbackground=ui.CLR_BORDER, highlightthickness=1,
        )
        box.grid(
            row=index // 3, column=column, sticky="nsew",
            padx=(0 if column == 0 else 4, 0 if column == 2 else 4),
            pady=(0 if index < 3 else 8, 0),
        )
        field_label = tk.Label(
            box, text=label, font=("Microsoft YaHei UI", 8, "bold"),
            fg=ui.CLR_TEXT_MUTED, bg=ui.CLR_PANEL,
            anchor="w", justify=tk.LEFT, wraplength=180,
        )
        field_label.pack(fill=tk.X, padx=9, pady=(8, 0))
        field_label.bind(
            "<Configure>",
            lambda event: field_label.configure(wraplength=max(1, event.width)),
        )
        if choices is not None:
            widget = ttk.Combobox(
                box, textvariable=variable, values=choices,
                state="readonly", width=15,
            )
            widget.pack(fill=tk.X, padx=9, pady=(4, 9), ipady=2)
        else:
            widget = tk.Entry(
                box, textvariable=variable, width=15, font=ui.FONT_MONO,
                fg=ui.CLR_TEXT_MAIN, bg=ui.CLR_PANEL,
                insertbackground=ui.CLR_TEXT_MAIN, relief="flat",
                highlightthickness=1, highlightbackground=ui.CLR_BORDER,
                highlightcolor=ui.COLOR_BLUE,
                disabledbackground=ui.CLR_PANEL,
                disabledforeground=ui.CLR_TEXT_MUTED,
            )
            widget.pack(fill=tk.X, padx=9, pady=(4, 9), ipady=4)
        return widget

    def _level_labels() -> tuple[str, ...]:
        return tuple(f"{area}-{sub}" for area in range(1, 6) for sub in range(1, 11))

    def _level_id(label: str) -> int:
        area_text, sub_text = label.split("-", 1)
        area = int(area_text)
        sub = int(sub_text)
        if not 1 <= area <= 5 or not 1 <= sub <= 10:
            raise ValueError("level must be Adventure 1-1 through 5-10")
        return (area - 1) * 10 + sub

    def _status_color(status: str) -> str:
        if status in ("RUNNING", "STARTING", "STOPPING"):
            return ui.COLOR_BLUE if status != "STOPPING" else ui.CLR_TEXT_MUTED
        if status in ("FINISHED", "PASS"):
            return ui.COLOR_GREEN
        if status in ("FAILED", "FAIL"):
            return ui.COLOR_RED
        if status == "PAUSED":
            return ui.LIVE_STATUS_COLORS.get("PAUSED", ui.CLR_TEXT_MUTED)
        return ui.CLR_TEXT_MUTED

    def _build_training_panel(self, parent):
        self._training_running = False
        self._training_pause_event = None
        self._training_stop_event = None
        self._training_controller = None
        self._training_checkpoint_path = None
        self._training_checkpoint_config = None
        self._training_loaded_config = None
        self._training_checkpoint_loading = False
        self._training_checkpoint_records = []
        self._training_checkpoint_evaluations = []
        self._training_evaluating = False
        self._training_history = _new_training_history()
        self._training_param_vars = {}
        self._training_last_random_step = 0
        self.var_trend_preset = tk.StringVar(value="全部 (E1-latest)")
        self.var_trend_start = tk.StringVar(value="1")
        self.var_trend_end = tk.StringVar(value="latest")

        panel = tk.Frame(parent, bg=ui.CLR_BG)
        panel.pack(fill=tk.BOTH, expand=True)
        self._training_panel = panel

        canvas = tk.Canvas(panel, bg=ui.CLR_BG, bd=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            panel,
            orient="vertical",
            style="Dark.Vertical.TScrollbar",
            command=canvas.yview,
        )
        content = tk.Frame(canvas, bg=ui.CLR_BG)
        window = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(window, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        outer = tk.Frame(
            content,
            bg=ui.CLR_PANEL,
            highlightbackground=ui.CLR_BORDER,
            highlightthickness=1,
        )
        outer.pack(fill=tk.X)

        top = tk.Frame(outer, bg=ui.CLR_PANEL)
        top.pack(fill=tk.X, padx=12, pady=(10, 8))
        tk.Label(
            top,
            text="Gym Training",
            font=ui.FONT_TITLE,
            fg=ui.CLR_TEXT_MAIN,
            bg=ui.CLR_PANEL,
        ).pack(side=tk.LEFT)
        self.lbl_training_status = tk.Label(
            top,
            text="READY",
            font=ui.FONT_BOLD,
            fg=ui.CLR_TEXT_MUTED,
            bg=ui.CLR_PANEL,
        )
        self.lbl_training_status.pack(side=tk.RIGHT)

        _, env_body = _card(outer, "关卡 / 选卡")

        from .mixed_level_panel import MixedLevelPanel
        self.mixed_level_panel = MixedLevelPanel(
            env_body, ui, on_change=lambda: _on_mixed_level_settings_changed(self),
        )
        self.mixed_level_panel.pack(fill=tk.X, pady=(0, 6))

        env_grid = tk.Frame(env_body, bg=ui.CLR_CARD)
        env_grid.pack(fill=tk.X)
        for column in range(3):
            env_grid.columnconfigure(column, weight=1, uniform="environment-parameter")
        self.var_training_level = tk.StringVar(value="1-4")
        self.var_training_ticks = tk.StringVar(value="150")
        self.var_training_max_episode_ticks = tk.StringVar(
            value=str(EnvConfig().max_episode_ticks)
        )
        self.var_training_playthrough = tk.StringVar(value="1")
        self.var_training_seed = tk.StringVar(value="0")
        self.var_training_seed_types = tk.StringVar(value="")

        self.cmb_training_level = _parameter_field(
            env_grid, 0, "关卡", self.var_training_level, _level_labels(),
        )
        self.cmb_training_playthrough = _parameter_field(
            env_grid, 1, "周目", self.var_training_playthrough, ("1", "2"),
        )

        self.training_deck_panel = DeckPanel(
            env_body, ui,
            lambda: (_level_id(self.var_training_level.get()), int(self.var_training_playthrough.get())),
            lambda panel: start_deck_test(self, panel),
            cards_var=self.var_training_seed_types, legacy=True)
        self.training_deck_panel.pack(fill=tk.X, pady=(0, 10))
        self.ent_training_seed_types = self.training_deck_panel.entry
        self.var_training_level.trace_add("write", lambda *_: self.training_deck_panel.refresh())
        self.var_training_playthrough.trace_add("write", lambda *_: self.training_deck_panel.refresh())

        _, reward_body = _card(outer, "奖励")
        reward_row = tk.Frame(reward_body, bg=ui.CLR_CARD)
        reward_row.pack(fill=tk.X)
        tk.Label(
            reward_row,
            text="WIN +1.0",
            font=ui.FONT_BOLD,
            fg=ui.COLOR_GREEN,
            bg=ui.CLR_CARD,
        ).pack(side=tk.LEFT, padx=(0, 18))
        tk.Label(
            reward_row,
            text="LOSS -1.0",
            font=ui.FONT_BOLD,
            fg=ui.COLOR_RED,
            bg=ui.CLR_CARD,
        ).pack(side=tk.LEFT, padx=(0, 18))
        self.var_training_mower_penalty = tk.StringVar(value="0.1")
        mower_box, self.ent_training_mower_penalty = _entry_field(
            reward_row, "Mower penalty / activation", self.var_training_mower_penalty, 10
        )
        mower_box.pack(side=tk.LEFT)
        self.lbl_training_reward_note = tk.Label(
            reward_body,
            text="Double DQN: mower activation is applied immediately on that transition.",
            font=("Microsoft YaHei UI", 8),
            fg=ui.CLR_TEXT_MUTED,
            bg=ui.CLR_CARD,
            anchor="w",
        )
        self.lbl_training_reward_note.pack(fill=tk.X, pady=(7, 0))

        _, params_body = _card(outer, "模型训练")
        trainer_body = params_body
        trainer_row = tk.Frame(trainer_body, bg=ui.CLR_CARD)
        trainer_row.pack(fill=tk.X)
        tk.Label(
            trainer_row,
            text="Algorithm",
            font=ui.FONT_BOLD,
            fg=ui.CLR_TEXT_MUTED,
            bg=ui.CLR_CARD,
        ).pack(side=tk.LEFT, padx=(0, 8))
        self.var_training_trainer = tk.StringVar(value="Double DQN")
        self.cmb_training_trainer = ttk.Combobox(
            trainer_row,
            textvariable=self.var_training_trainer,
            values=tuple(spec.label for spec in all_trainer_specs()),
            state="readonly",
            width=20,
        )
        self.cmb_training_trainer.pack(side=tk.LEFT, padx=(0, 12))
        self.cmb_training_trainer.bind(
            "<<ComboboxSelected>>", lambda _e: self._training_trainer_changed()
        )
        self.lbl_training_trainer_state = tk.Label(
            trainer_row,
            text="AVAILABLE",
            font=ui.FONT_BOLD,
            fg=ui.COLOR_GREEN,
            bg=ui.CLR_CARD,
        )
        self.lbl_training_trainer_state.pack(side=tk.RIGHT)
        self.lbl_training_trainer_description = tk.Label(
            trainer_body,
            text="",
            font=ui.FONT_MAIN,
            fg=ui.CLR_TEXT_MUTED,
            bg=ui.CLR_CARD,
            anchor="w",
            justify=tk.LEFT,
            wraplength=760,
        )
        self.lbl_training_trainer_description.pack(fill=tk.X, pady=(7, 0))

        self.training_params_frame = tk.Frame(params_body, bg=ui.CLR_CARD)
        self.training_params_frame.pack(fill=tk.X)
        self.training_runtime_card, runtime_body = _card(outer, "PvZ 运行参数")
        runtime_common = tk.Frame(runtime_body, bg=ui.CLR_CARD)
        runtime_common.pack(fill=tk.X, pady=(0, 8))
        for col in range(3):
            runtime_common.columnconfigure(col, weight=1, uniform="runtime-common")
        self.ent_training_ticks = _parameter_field(
            runtime_common, 0, "Ticks / step", self.var_training_ticks,
        )
        self.ent_training_max_episode_ticks = _parameter_field(
            runtime_common, 1, "单局 tick 上限", self.var_training_max_episode_ticks,
        )
        self.lbl_training_tick_limit_note = tk.Label(
            runtime_body, font=("Microsoft YaHei UI", 8),
            fg=ui.CLR_TEXT_MUTED, bg=ui.CLR_CARD,
            anchor="w", justify=tk.LEFT,
        )
        self.lbl_training_tick_limit_note.pack(fill=tk.X, pady=(7, 0))
        self.lbl_training_tick_limit_note.bind(
            "<Configure>",
            lambda event: self.lbl_training_tick_limit_note.configure(
                wraplength=max(1, event.width)
            ),
        )

        def update_tick_limit_note(*_args):
            try:
                ticks = int(self.var_training_max_episode_ticks.get())
                if ticks < 0:
                    raise ValueError
            except ValueError:
                note = "窗口上限请输入非负整数 tick；0 表示不限时。"
            else:
                note = (
                    f"窗口时长：{ticks / 6000:g} 分钟游戏时间；达到上限后重开新局。0 表示不限时。"
                    if ticks else "窗口不限时；设置大于 0 的 tick 上限可在到时后重开新局。"
                )
            self.lbl_training_tick_limit_note.configure(text=note)

        self.var_training_max_episode_ticks.trace_add("write", update_tick_limit_note)
        update_tick_limit_note()

        self.training_runtime_frame = tk.Frame(runtime_body, bg=ui.CLR_CARD)
        self.training_runtime_frame.pack(fill=tk.X)
        for col in range(3):
            self.training_runtime_frame.columnconfigure(
                col,
                weight=1,
                uniform="runtime-parameter",
            )

        self.training_checkpoint_card, checkpoint_body = _card(outer, "Checkpoint")
        checkpoint_top = tk.Frame(checkpoint_body, bg=ui.CLR_CARD)
        checkpoint_top.pack(fill=tk.X)
        self.lbl_training_checkpoint = tk.Label(
            checkpoint_top,
            text="New training (no checkpoint loaded)",
            font=ui.FONT_MONO,
            fg=ui.CLR_TEXT_MUTED,
            bg=ui.CLR_CARD,
            anchor="w",
        )
        self.lbl_training_checkpoint.pack(side=tk.LEFT, fill=tk.X, expand=True)
        checkpoint_actions = tk.Frame(checkpoint_body, bg=ui.CLR_CARD)
        checkpoint_actions.pack(fill=tk.X, pady=(7, 0))
        for col in range(3):
            checkpoint_actions.columnconfigure(col, weight=1, uniform="checkpoint-action")
        self.btn_training_load_checkpoint = ui.CyberButton(
            checkpoint_actions,
            "Load checkpoint",
            self._load_training_checkpoint,
            bg_color=ui.CLR_PANEL,
            fg_color=ui.CLR_TEXT_MAIN,
            height=32,
        )
        self.btn_training_load_checkpoint.grid(row=0, column=0, padx=(0, 3), sticky="ew")
        self.btn_training_new_checkpoint = ui.CyberButton(
            checkpoint_actions,
            "New training",
            self._clear_training_checkpoint,
            bg_color=ui.CLR_PANEL,
            fg_color=ui.CLR_TEXT_MAIN,
            height=32,
        )
        self.btn_training_new_checkpoint.grid(row=0, column=1, padx=3, sticky="ew")
        self.btn_training_save_checkpoint = ui.CyberButton(
            checkpoint_actions,
            "Save checkpoint",
            self._save_training_checkpoint,
            bg_color=ui.CLR_PANEL,
            fg_color=ui.COLOR_BLUE,
            height=32,
        )
        self.btn_training_save_checkpoint.grid(row=0, column=2, padx=(3, 0), sticky="ew")

        # Evaluation gets its own row so it remains visible on narrow windows.
        checkpoint_evaluation = tk.Frame(checkpoint_body, bg=ui.CLR_CARD)
        checkpoint_evaluation.pack(fill=tk.X, pady=(7, 0))
        self.var_training_eval_episodes = tk.StringVar(value="10")
        eval_box, _ = _entry_field(
            checkpoint_evaluation,
            "Test games",
            self.var_training_eval_episodes,
            7,
        )
        eval_box.pack(side=tk.LEFT, padx=(0, 8))
        self.btn_training_test_checkpoint = ui.CyberButton(
            checkpoint_evaluation,
            "Test loaded model",
            self._test_training_checkpoint,
            bg_color=ui.CLR_PANEL,
            fg_color=ui.COLOR_GREEN,
            height=32,
        )
        self.btn_training_test_checkpoint.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(
            checkpoint_body,
            text=(
                "Load shows the score saved in the .pt; Test loaded model runs greedy "
                "games (no exploration) to measure its actual win rate and score."
            ),
            font=("Microsoft YaHei UI", 8),
            fg=ui.CLR_TEXT_MUTED,
            bg=ui.CLR_CARD,
            anchor="w",
        ).pack(fill=tk.X, pady=(7, 0))

        controls = tk.Frame(outer, bg=ui.CLR_PANEL)
        controls.pack(fill=tk.X, padx=12, pady=(2, 8))
        for col in range(5):
            controls.columnconfigure(col, weight=1)
        self.btn_training_run = ui.CyberButton(
            controls,
            "Start",
            self._start_training,
            bg_color=ui.COLOR_BLUE,
            fg_color="#ffffff",
            height=36,
        )
        self.btn_training_run.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        self.btn_training_pause = ui.CyberButton(
            controls,
            "Pause",
            self._pause_training,
            bg_color=ui.CLR_CARD,
            fg_color=ui.CLR_TEXT_MAIN,
            height=36,
        )
        self.btn_training_pause.grid(row=0, column=1, padx=4, sticky="ew")
        self.btn_training_continue = ui.CyberButton(
            controls,
            "Continue",
            self._continue_training,
            bg_color=ui.CLR_CARD,
            fg_color=ui.CLR_TEXT_MAIN,
            height=36,
        )
        self.btn_training_continue.grid(row=0, column=2, padx=4, sticky="ew")
        self.btn_training_stop = ui.CyberButton(
            controls,
            "Stop",
            self._stop_training,
            bg_color=ui.CLR_CARD,
            fg_color=ui.COLOR_RED,
            height=36,
        )
        self.btn_training_stop.grid(row=0, column=4, padx=(4, 0), sticky="ew")
        self.btn_training_reset_game = ui.CyberButton(
            controls, "Reset Game", self._reset_training_game,
            bg_color=ui.CLR_CARD, fg_color=ui.CLR_TEXT_MAIN, height=36)
        self.btn_training_reset_game.grid(row=0, column=3, padx=4, sticky="ew")

        self.lbl_training_progress = tk.Label(
            outer,
            text="READY",
            font=ui.FONT_BOLD,
            fg=ui.CLR_TEXT_MUTED,
            bg=ui.CLR_PANEL,
            anchor="w",
        )
        self.lbl_training_progress.pack(fill=tk.X, padx=12, pady=(0, 7))

        metrics = tk.Frame(outer, bg=ui.CLR_PANEL)
        metrics.pack(fill=tk.X, padx=12, pady=(0, 8))
        self._training_metric_labels = {}
        metric_specs = (
            ("step", "Step", "0"),
            ("episode", "Episode", "0"),
            ("record", "Wins / Losses", "0 / 0"),
            ("win_rate", "Win rate", "0.0%"),
            ("reward", "Score / reward (mean)", "0.000"),
            ("loss", "Loss", "-"),
            ("epsilon", "Epsilon", "-"),
            ("replay", "Replay", "-"),
            ("mowers", "Mowers", "0"),
            ("invalid", "Invalid", "0"),
            ("ticks", "Ticks actual (mean ± std)", "-"),
            ("long_ticks", "Long tick jumps (>2x)", "-"),
        )
        for col in range(3):
            metrics.columnconfigure(col, weight=1)
        for index, (key, title, value) in enumerate(metric_specs):
            card = tk.Frame(
                metrics,
                bg=ui.CLR_CARD,
                highlightbackground=ui.CLR_BORDER,
                highlightthickness=1,
            )
            card.grid(
                row=index // 3,
                column=index % 3,
                sticky="ew",
                padx=(0 if index % 3 == 0 else 3, 0),
                pady=(0 if index < 3 else 3, 0),
            )
            tk.Label(
                card,
                text=title,
                font=("Microsoft YaHei UI", 8),
                fg=ui.CLR_TEXT_MUTED,
                bg=ui.CLR_CARD,
            ).pack(pady=(5, 0))
            label = tk.Label(
                card,
                text=value,
                font=ui.FONT_BOLD,
                fg=ui.COLOR_BLUE,
                bg=ui.CLR_CARD,
            )
            label.pack(pady=(1, 5))
            self._training_metric_labels[key] = label

        _, chart_body = _card(outer, "Live trend · 训练走势")

        # 走势图范围选择工具栏
        trend_ctrl = tk.Frame(chart_body, bg=ui.CLR_CARD)
        trend_ctrl.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            trend_ctrl, text="预设:", font=("Microsoft YaHei UI", 8),
            fg=ui.CLR_TEXT_MUTED, bg=ui.CLR_CARD
        ).pack(side=tk.LEFT, padx=(0, 4))

        self.cmb_trend_preset = ttk.Combobox(
            trend_ctrl,
            textvariable=self.var_trend_preset,
            values=["全部 (E1-latest)", "最近 100 局", "最近 200 局", "最近 400 局", "最近 800 局", "自定义"],
            width=14,
            state="readonly"
        )
        self.cmb_trend_preset.pack(side=tk.LEFT, padx=(0, 10))
        self.cmb_trend_preset.bind("<<ComboboxSelected>>", lambda _e: self._on_trend_preset_changed())

        tk.Label(
            trend_ctrl, text="范围:", font=("Microsoft YaHei UI", 8),
            fg=ui.CLR_TEXT_MUTED, bg=ui.CLR_CARD
        ).pack(side=tk.LEFT, padx=(0, 2))

        tk.Label(
            trend_ctrl, text="E", font=("Consolas", 8, "bold"),
            fg=ui.COLOR_BLUE, bg=ui.CLR_CARD
        ).pack(side=tk.LEFT)

        self.entry_trend_start = tk.Entry(
            trend_ctrl,
            textvariable=self.var_trend_start,
            width=6,
            font=ui.FONT_MONO,
            fg=ui.CLR_TEXT_MAIN,
            bg=ui.CLR_PANEL,
            relief="flat",
            highlightthickness=1,
            highlightbackground=ui.CLR_BORDER,
            highlightcolor=ui.COLOR_BLUE,
        )
        self.entry_trend_start.pack(side=tk.LEFT, padx=(1, 4))
        self.entry_trend_start.bind("<Return>", lambda _e: self._apply_trend_custom_range())
        self.entry_trend_start.bind("<FocusOut>", lambda _e: self._apply_trend_custom_range())

        tk.Label(
            trend_ctrl, text="至  E", font=("Consolas", 8, "bold"),
            fg=ui.COLOR_BLUE, bg=ui.CLR_CARD
        ).pack(side=tk.LEFT)

        self.entry_trend_end = tk.Entry(
            trend_ctrl,
            textvariable=self.var_trend_end,
            width=8,
            font=ui.FONT_MONO,
            fg=ui.CLR_TEXT_MAIN,
            bg=ui.CLR_PANEL,
            relief="flat",
            highlightthickness=1,
            highlightbackground=ui.CLR_BORDER,
            highlightcolor=ui.COLOR_BLUE,
        )
        self.entry_trend_end.pack(side=tk.LEFT, padx=(1, 8))
        self.entry_trend_end.bind("<Return>", lambda _e: self._apply_trend_custom_range())
        self.entry_trend_end.bind("<FocusOut>", lambda _e: self._apply_trend_custom_range())

        self.btn_trend_apply = tk.Button(
            trend_ctrl,
            text="应用",
            font=("Microsoft YaHei UI", 8),
            fg=ui.CLR_TEXT_MAIN,
            bg=ui.CLR_PANEL,
            relief="flat",
            padx=8,
            command=self._apply_trend_custom_range
        )
        self.btn_trend_apply.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_trend_reset = tk.Button(
            trend_ctrl,
            text="重置",
            font=("Microsoft YaHei UI", 8),
            fg=ui.CLR_TEXT_MUTED,
            bg=ui.CLR_PANEL,
            relief="flat",
            padx=6,
            command=self._reset_trend_range
        )
        self.btn_trend_reset.pack(side=tk.LEFT)

        self.training_chart = tk.Canvas(
            chart_body,
            height=318,
            bg=ui.CLR_PANEL,
            highlightthickness=0,
        )
        self.training_chart.pack(fill=tk.X)
        self.training_chart.bind("<Configure>", lambda _e: self._draw_training_chart())

        self._training_trainer_changed()
        self._set_training_controls(False)
        self._draw_training_chart()

    def _selected_training_spec(self):
        return get_trainer_spec(trainer_key_from_label(self.var_training_trainer.get()))

    def _training_trainer_changed(self):
        if self._training_running:
            return
        spec = self._selected_training_spec()
        self.lbl_training_trainer_description.config(
            text=spec.description if spec.available else f"{spec.description}  {spec.unavailable_reason}"
        )
        self.lbl_training_trainer_state.config(
            text="AVAILABLE" if spec.available else "RESERVED",
            fg=ui.COLOR_GREEN if spec.available else ui.CLR_TEXT_MUTED,
        )

        for child in self.training_params_frame.winfo_children():
            child.destroy()
        self._training_param_vars = {}
        self._training_param_widgets = {}
        from .parameter_sections import ParameterSection
        common_keys = {"total_steps", "episodes", "max_steps", "learning_rate", "batch_size"}
        exploration_keys = {"epsilon_start", "epsilon_end", "epsilon_decay_steps",
                            "exploration_mode", "exploration_ratio"}
        evaluation_keys = {"evaluation_mode", "evaluation_per_level", "evaluation_random_episodes"}
        sampling_keys = {"level_sampling_mode", "adaptive_level_sampling", "level_sampling_window", "level_sampling_update_every",
                         "level_sampling_uniform_mix", "level_sampling_max_weight"}
        groups = []
        for title, keys in (("训练设置", common_keys), ("探索设置", exploration_keys),
                            ("评估设置", evaluation_keys), ("关卡抽样", sampling_keys)):
            if not any(p.key in keys for p in spec.parameters):
                continue
            tk.Label(self.training_params_frame, text=title, font=ui.FONT_BOLD,
                     fg=ui.CLR_TEXT_MUTED, bg=ui.CLR_CARD, anchor="w").pack(
                         fill=tk.X, pady=(10, 6))
            frame = tk.Frame(self.training_params_frame, bg=ui.CLR_CARD)
            frame.pack(fill=tk.X)
            for col in range(3):
                frame.columnconfigure(col, weight=1, uniform="trainer-parameter")
            groups.append((frame, [p for p in spec.parameters if p.key in keys]))
        self.training_model_advanced = ParameterSection(
            self.training_params_frame, ui, "高级参数")
        self.training_model_advanced.pack(fill=tk.X, pady=(10, 0))
        groups.append((self.training_model_advanced.body,
                       [p for p in spec.parameters if p.key not in common_keys | exploration_keys | evaluation_keys | sampling_keys]))
        for frame, parameters in groups:
            for index, parameter in enumerate(parameters):
                default_val = parameter.default
                if parameter.key == "evaluation_mode":
                    is_multi = bool(getattr(self, "mixed_level_panel", None) and self.mixed_level_panel.is_mixed_mode())
                    pool = self.mixed_level_panel.get_level_pool() if is_multi else None
                    default_val = "per_level" if (is_multi and pool and len(pool) > 1) else "random"
                variable = tk.StringVar(value=str(default_val))
                self._training_param_vars[parameter.key] = variable
                widget = _parameter_field(
                    frame, index, parameter.label, variable,
                    parameter.choices if parameter.kind in ("choice", "bool") else None,
                )
                if frame is self.training_model_advanced.body:
                    self.training_model_advanced.watch(variable, parameter.default)
                if not spec.available:
                    widget.config(state="disabled")
                self._training_param_widgets[parameter.key] = widget

        self.lbl_training_level_sampling = tk.Label(self.training_params_frame,
            text="近期胜率与抽样概率将在训练开始后显示", fg=ui.CLR_TEXT_MUTED,
            bg=ui.CLR_CARD, anchor="w", justify=tk.LEFT, wraplength=500)
        if spec.key == "double_dqn":
            self.lbl_training_level_sampling.pack(fill=tk.X, pady=(6, 0))

        if "exploration_mode" in self._training_param_vars:
            ratio_box = self._training_param_widgets["exploration_ratio"].master
            self.lbl_training_exploration_note = tk.Label(
                ratio_box, text="", font=("Microsoft YaHei UI", 8),
                fg=ui.CLR_TEXT_MUTED, bg=ui.CLR_PANEL, anchor="w", justify=tk.LEFT,
                wraplength=180,
            )
            self.lbl_training_exploration_note.pack(fill=tk.X, padx=9, pady=(0, 8))
            self.lbl_training_exploration_note.bind(
                "<Configure>", lambda event: self.lbl_training_exploration_note.configure(
                    wraplength=max(1, event.width)),
            )
            for key in ("exploration_mode", "exploration_ratio"):
                self._training_param_vars[key].trace_add(
                    "write", lambda *_: self._refresh_training_exploration_controls(),
                )
        if "replay_scheme" in self._training_param_vars:
            self._training_param_vars["replay_scheme"].trace_add(
                "write", lambda *_: self._refresh_training_replay_controls(),
            )
        if "auto_train_every" in self._training_param_vars:
            self._training_param_vars["auto_train_every"].trace_add(
                "write", lambda *_: self._refresh_training_auto_train_controls(),
            )
        self._evaluation_mode_user_modified = False
        if "evaluation_mode" in self._training_param_vars:
            widget = self._training_param_widgets.get("evaluation_mode")
            if widget is not None and hasattr(widget, "bind"):
                widget.bind("<<ComboboxSelected>>", lambda _: setattr(self, "_evaluation_mode_user_modified", True))
            self._training_param_vars["evaluation_mode"].trace_add(
                "write", lambda *_: self._refresh_training_evaluation_controls(),
            )
        self._refresh_training_evaluation_controls()

        dqn = spec.key == "double_dqn"
        self.ent_training_mower_penalty.config(state="normal" if dqn else "disabled")
        self.lbl_training_reward_note.config(
            text=(
                "Double DQN: mower activation is applied immediately on that transition."
                if dqn
                else "Terminal reward remains a shared training contract; mower shaping is inactive for this trainer."
            )
        )
        self._set_training_checkpoint_controls(dqn)
        self._set_training_controls(False)

    def _set_training_checkpoint_controls(self, dqn: bool):
        self.btn_training_load_checkpoint.set_disabled((not dqn) or self._training_running)
        self.btn_training_new_checkpoint.set_disabled((not dqn) or self._training_running)
        self.btn_training_test_checkpoint.set_disabled(
            (not dqn)
            or self._training_running
            or not bool(self._training_checkpoint_path)
        )
        self.btn_training_save_checkpoint.set_disabled(
            (not dqn) or self._training_controller is None
        )

    def _set_training_controls(self, running: bool, paused: bool = False):
        if hasattr(self, "training_deck_panel"):
            self.training_deck_panel.set_busy(running)
        try:
            spec = self._selected_training_spec()
            available = spec.available
        except Exception:
            available = False
        self.btn_training_run.set_disabled(running or not available)
        self.btn_training_pause.set_disabled((not running) or paused)
        self.btn_training_continue.set_disabled((not running) or (not paused))
        self.btn_training_stop.set_disabled(not running)
        if hasattr(self, "btn_training_reset_game"):
            self.btn_training_reset_game.set_disabled(
                not (running and paused and self._training_controller is not None)
                or bool(getattr(self._training_controller, "_evaluation_active", False)))
        if hasattr(self, "cmb_training_trainer"):
            self.cmb_training_trainer.config(state="disabled" if running else "readonly")
        if hasattr(self, "cmb_training_level"):
            self.cmb_training_level.config(state="disabled" if running else "readonly")
        if hasattr(self, "cmb_training_playthrough"):
            self.cmb_training_playthrough.config(state="disabled" if running else "readonly")
        if hasattr(self, "ent_training_max_episode_ticks"):
            self.ent_training_max_episode_ticks.config(state="disabled" if running else "normal")
        pausable_keys = {
            "exploration_mode", "exploration_ratio", "exploration_type_weights",
            "replay_scheme", "per_alpha", "per_beta_start",
        }
        for parameter in self._selected_training_spec().parameters:
            widget = self._training_param_widgets.get(parameter.key)
            if widget is not None:
                # 暂停状态下放行探索策略与回放机制参数编辑
                can_edit = (not running) or (paused and parameter.key in pausable_keys)
                widget.config(state=("readonly" if parameter.kind in ("choice", "bool") else "normal")
                              if can_edit and available else "disabled")
        self._refresh_training_exploration_controls(running, paused=paused)
        self._refresh_training_replay_controls(running, paused=paused)
        self._refresh_training_auto_train_controls(running, paused=paused)
        self._refresh_training_evaluation_controls(running, paused=paused)
        if hasattr(self, "btn_training_load_checkpoint"):
            self._set_training_checkpoint_controls(
                self._selected_training_spec().key == "double_dqn"
            )
        if hasattr(self, "training_deck_panel"):
            from .checkpoint_load_mode import _is_resume_label
            resume_locked = (self._selected_training_spec().key == "double_dqn"
                             and _is_resume_label(self.var_training_checkpoint_load_mode.get())
                             and bool(self._training_checkpoint_path))
            self.training_deck_panel.set_busy(running or resume_locked or self._training_checkpoint_loading)

    def _refresh_training_evaluation_controls(self, running=None, paused=None):
        variables = self._training_param_vars
        if "evaluation_mode" not in variables:
            return
        mode = variables["evaluation_mode"].get().strip().lower()
        running = self._training_running if running is None else running
        try:
            available = self._selected_training_spec().available
        except Exception:
            available = True
        can_edit = (not running) and available

        if "evaluation_per_level" in self._training_param_widgets:
            state = "normal" if (can_edit and mode == "per_level") else "disabled"
            self._training_param_widgets["evaluation_per_level"].config(state=state)
        if "evaluation_random_episodes" in self._training_param_widgets:
            state = "normal" if (can_edit and mode == "random") else "disabled"
            self._training_param_widgets["evaluation_random_episodes"].config(state=state)

    def _refresh_training_auto_train_controls(self, running=None, paused=None):
        variables = self._training_param_vars
        if "auto_train_every" not in variables:
            return
        raw_val = variables["auto_train_every"].get()
        auto_enabled = str(raw_val).strip().lower() in ("true", "1")
        running = self._training_running if running is None else running
        locked = running
        for key in ("auto_train_high_watermark", "auto_train_low_watermark"):
            if key in self._training_param_widgets:
                self._training_param_widgets[key].config(
                    state="disabled" if (locked or not auto_enabled) else "normal"
                )

    def _refresh_training_replay_controls(self, running=None, paused=None):
        variables = self._training_param_vars
        if "replay_scheme" not in variables:
            return
        scheme = variables["replay_scheme"].get()
        running = self._training_running if running is None else running
        if paused is None:
            paused = (
                self._training_pause_event is not None
                and not self._training_pause_event.is_set()
            ) or (
                self._training_controller is not None
                and getattr(self._training_controller, "pause_event", None) is not None
                and not self._training_controller.pause_event.is_set()
            )
        locked = running and not paused
        if "replay_scheme" in self._training_param_widgets:
            self._training_param_widgets["replay_scheme"].config(
                state="disabled" if locked else "readonly"
            )
        per_active = (scheme == "per")
        for key in ("per_alpha", "per_beta_start"):
            if key in self._training_param_widgets:
                self._training_param_widgets[key].config(
                    state="normal" if (not locked and per_active) else "disabled"
                )

    def _refresh_training_exploration_controls(self, running=None, paused=None):
        variables = self._training_param_vars
        if "exploration_mode" not in variables:
            return
        modes = {label: key for key, label in EXPLORATION_MODE_LABELS.items()}
        mode = modes.get(variables["exploration_mode"].get(), "epsilon_random")
        running = self._training_running if running is None else running
        if paused is None:
            paused = (
                self._training_pause_event is not None
                and not self._training_pause_event.is_set()
            ) or (
                self._training_controller is not None
                and getattr(self._training_controller, "pause_event", None) is not None
                and not self._training_controller.pause_event.is_set()
            )
        # 暂停状态下解锁编辑，非暂停运行态则锁定
        locked = running and not paused
        if "exploration_mode" in self._training_param_widgets:
            self._training_param_widgets["exploration_mode"].config(
                state="disabled" if locked else "readonly"
            )
        if "exploration_ratio" in self._training_param_widgets:
            self._training_param_widgets["exploration_ratio"].config(
                state="disabled" if locked or mode == "epsilon_random" else "normal"
            )
        if "exploration_type_weights" in self._training_param_widgets:
            self._training_param_widgets["exploration_type_weights"].config(
                state="disabled" if locked else "normal"
            )
        if mode == "epsilon_random":
            text = "探索内：100% 纯随机；比例不生效"
        else:
            try:
                fraction = q_guided_fraction(mode, parse_exploration_ratio(variables["exploration_ratio"].get()))
                random_label = "广泛" if mode == "q_guided_mixed" else "邻格"
                text = f"探索内：Q {fraction:.0%} / {random_label} {1 - fraction:.0%}"
            except ValueError:
                text = "填写 Q:随机，例如 50:50；不可为 0:0"
        self.lbl_training_exploration_note.config(text=text)

    def _parse_common_training_config(self):
        level_id = _level_id(self.var_training_level.get())
        ticks = int(self.var_training_ticks.get())
        max_ticks = int(self.var_training_max_episode_ticks.get())
        if max_ticks < 0:
            raise ValueError("Max ticks / window must be non-negative")
        self._training_max_episode_ticks = max_ticks or None
        playthrough = int(self.var_training_playthrough.get())
        seed = int(self.var_training_seed.get())
        mower_penalty = float(self.var_training_mower_penalty.get())
        if ticks <= 0:
            raise ValueError("Ticks / step must be positive")
        if playthrough not in (1, 2):
            raise ValueError("Playthrough must be 1 or 2")
        if mower_penalty < 0:
            raise ValueError("Mower penalty must be non-negative")
        self._training_seed_selection_mode, seed_types = self.training_deck_panel.settings()
        return level_id, ticks, playthrough, seed, mower_penalty, seed_types

    def _parse_trainer_parameters(self):
        spec = self._selected_training_spec()
        values = {}
        for parameter in spec.parameters:
            values[parameter.key] = parameter.parse(
                self._training_param_vars[parameter.key].get()
            )
        if spec.key == "double_dqn":
            mode = next(key for key, label in EXPLORATION_MODE_LABELS.items()
                        if label == values["exploration_mode"])
            if mode == "epsilon_random":
                values["exploration_ratio"] = format_exploration_ratio(DEFAULT_EXPLORATION_RATIO)
            else:
                parse_exploration_ratio(values["exploration_ratio"])
        return spec, values

    def _on_mixed_level_settings_changed(self) -> None:
        if not hasattr(self, "mixed_level_panel"):
            return
        is_mixed = self.mixed_level_panel.is_mixed_mode()
        pool = self.mixed_level_panel.get_level_pool() if is_mixed else None
        has_multiple_levels = bool(pool and len(pool) > 1) if is_mixed else False

        # 联动更新评估模式：单关默认 random，多关默认 per_level（若用户未修改且非关闭状态）
        if hasattr(self, "_training_param_vars") and "evaluation_mode" in self._training_param_vars:
            current_mode = self._training_param_vars["evaluation_mode"].get().strip().lower()
            expected_mode = "per_level" if has_multiple_levels else "random"
            if not getattr(self, "_evaluation_mode_user_modified", False) and current_mode != "off":
                self._training_param_vars["evaluation_mode"].set(expected_mode)

        if is_mixed:
            if pool:
                first_lvl, first_p = pool[0]
                try:
                    from train.curriculum import format_level_label
                except ImportError:
                    from train.curriculum import format_level_label
                label = format_level_label(first_lvl)
                area_sub = tuple(f"{a}-{s}" for a in range(1, 6) for s in range(1, 11))
                if label in area_sub:
                    self.var_training_level.set(label)
                self.var_training_playthrough.set(str(first_p))

    def _start_training(self):
        if getattr(self, "_preparing_training_save", False):
            self.log("WARN", "[TRAIN] 等待训练存档准备完成后再开始训练。")
            return
        if self._training_running:
            self.log("WARN", "[TRAIN] a task is already running")
            return
        if self._training_checkpoint_loading:
            self.log("WARN", "[TRAIN] 等待历史配置读取完成后再开始训练。")
            return
        if getattr(self, "_live_running", False) or getattr(self, "_static_test_running", False):
            self.log("WARN", "[TRAIN] Adventure LIVE or Native Test is running")
            return
        try:
            level_id, ticks, playthrough, seed, mower_penalty, seed_types = (
                self._parse_common_training_config()
            )
            spec, params = self._parse_trainer_parameters()
            if not spec.available:
                raise RuntimeError(spec.unavailable_reason)
            if spec.key == 'double_dqn':
                game_runtime, training_runtime, turbo_policy = runtime_settings(self,
                    env=self._training_loaded_config.env if self._training_loaded_config else None,
                    runtime=self._training_loaded_config.runtime if self._training_loaded_config else None)
            level_pool = (
                self.mixed_level_panel.get_level_pool()
                if getattr(self, "mixed_level_panel", None) and self.mixed_level_panel.is_mixed_mode()
                else None
            )
            if getattr(self, "mixed_level_panel", None) and self.mixed_level_panel.is_mixed_mode():
                if not level_pool:
                    raise ValueError("混合关卡模式下必须至少勾选一个有效场景！")
                effective_seed_selection_mode = self.mixed_level_panel.get_seed_selection_mode()
                effective_level_id = level_pool[0][0]
                effective_playthrough = level_pool[0][1]
            else:
                effective_seed_selection_mode = self._training_seed_selection_mode
                effective_level_id = level_id
                effective_playthrough = playthrough

            if spec.key == "double_dqn" and self._training_loaded_config is not None:
                from .checkpoint_load_mode import _is_resume_label
                if (self._training_checkpoint_path
                        and _is_resume_label(self.var_training_checkpoint_load_mode.get())):
                    saved = self._training_loaded_config
                    # 关键保障：若历史训练包含关卡池，接续训练坚决不能悄悄退回单关
                    if getattr(saved.env, "level_pool", None) and not level_pool:
                        if hasattr(self, "mixed_level_panel"):
                            self.mixed_level_panel.set_from_level_pool(
                                saved.env.level_pool,
                                seed_selection_mode=saved.env.seed_selection_mode,
                            )
                            level_pool = self.mixed_level_panel.get_level_pool()
                        else:
                            level_pool = saved.env.level_pool
                        if level_pool:
                            effective_seed_selection_mode = saved.env.seed_selection_mode
                            effective_level_id = level_pool[0][0]
                            effective_playthrough = level_pool[0][1]
                            self.log(
                                "WARN",
                                f"[TRAIN:DQN] 接续训练已自动恢复历史关卡池 ({len(level_pool)} 关)，防止退回单关。",
                            )

                    requested = replace(
                        saved,
                        seed=seed,
                        env=replace(
                            saved.env,
                            level_id=effective_level_id,
                            playthrough=effective_playthrough,
                            level_pool=level_pool,
                        ),
                    )
                    validate_resume_config(requested, self._training_checkpoint_config or saved)
                    if seed_types != getattr(self, "_training_loaded_seed_types", None):
                        raise ValueError("接续训练不能修改卡组；请使用权重微调模式。")

            if spec.key == "double_dqn":
                env_config = EnvConfig(
                    level_id=effective_level_id,
                    playthrough=effective_playthrough,
                    ticks_per_step=ticks,
                    max_episode_ticks=self._training_max_episode_ticks,
                    seed_selection_mode=effective_seed_selection_mode,
                    seed_priority=seed_types if effective_seed_selection_mode == "custom" else None,
                    level_pool=level_pool,
                    runtime=game_runtime,
                    turbo=turbo_policy,
                )
                if level_pool:
                    self.log(
                        "INFO",
                        f"[TRAIN:DQN] 模式: 混合场景训练 (共 {len(level_pool)} 关, 选卡策略: {effective_seed_selection_mode})",
                    )
                config = DoubleDQNConfig(
                    env=env_config,
                    runtime=training_runtime,
                    total_steps=int(params["total_steps"]),
                    replay_capacity=int(params["replay_capacity"]),
                    learning_starts=int(params["learning_starts"]),
                    batch_size=int(params["batch_size"]),
                    selected_action_forward=bool(params["selected_action_forward"]),
                    boss_projectiles=True,
                    n_step=int(params["n_step"]),
                    gamma=float(params["gamma"]),
                    learning_rate=float(params["learning_rate"]),
                    train_every=int(params["train_every"]),
                    auto_train_every=bool(params.get("auto_train_every", True)),
                    auto_train_high_watermark=int(params.get("auto_train_high_watermark", 16)),
                    auto_train_low_watermark=int(params.get("auto_train_low_watermark", 4)),
                    training_architecture=str(params["training_architecture"]),
                    policy_sync_interval=int(params["policy_sync_interval"]),
                    max_pending_updates=int(params["max_pending_updates"]),
                    gradient_steps=int(params["gradient_steps"]),
                    evaluation_mode=str(params.get("evaluation_mode", "per_level" if (level_pool and len(level_pool) > 1) else "random")),
                    evaluation_per_level=int(params.get("evaluation_per_level", 4)),
                    evaluation_random_episodes=int(params.get("evaluation_random_episodes", 20)),
                    level_sampling_mode=str(params.get("level_sampling_mode", "squared_failure")),
                    adaptive_level_sampling=bool(params.get("adaptive_level_sampling", True)),
                    level_sampling_window=int(params.get("level_sampling_window", 40)),
                    level_sampling_update_every=int(params.get("level_sampling_update_every", 20)),
                    level_sampling_uniform_mix=float(params.get("level_sampling_uniform_mix", 0.05)),
                    level_sampling_max_weight=float(params.get("level_sampling_max_weight", 4)),
                    target_update_interval=int(params["target_update_interval"]),
                    epsilon_start=float(params["epsilon_start"]),
                    epsilon_end=float(params["epsilon_end"]),
                    epsilon_decay_steps=int(params["epsilon_decay_steps"]),
                    exploration_mode=next(key for key, label in EXPLORATION_MODE_LABELS.items()
                                          if label == params["exploration_mode"]),
                    exploration_ratio=parse_exploration_ratio(params["exploration_ratio"]),
                    exploration_type_weights=parse_action_family_weights(
                        str(params["exploration_type_weights"])
                    ),
                    mower_penalty=mower_penalty,
                    autosave_every=int(params["autosave_every"]),
                    checkpoint_every=int(params["checkpoint_every"]),
                    seed=seed,
                    device=str(params["device"]),
                    replay_scheme=str(params.get("replay_scheme", "per")),
                    per_alpha=float(params.get("per_alpha", 0.6)),
                    per_beta_start=float(params.get("per_beta_start", 0.4)),
                )
                saved_config = self._training_loaded_config if self._training_checkpoint_path else None
                if saved_config is not None:
                    # Start from the full source, preserving fields without GUI controls.
                    updates = {parameter.key: getattr(config, parameter.key) for parameter in spec.parameters}
                    config = replace(saved_config, **updates, env=env_config, runtime=training_runtime, seed=seed,
                                     mower_penalty=mower_penalty)
                run_token = getattr(self, "_training_run_token", None)
                from .checkpoint_load_mode import _mode_key
                self._training_controller = DoubleDQNTrainerController(
                    config,
                    dll_path=getattr(self.client, "dll_path", None),
                    seed_types=seed_types,
                    checkpoint_path=self._training_checkpoint_path,
                    checkpoint_load_mode=_mode_key(self.var_training_checkpoint_load_mode.get()),
                    on_event=lambda event, data, token=run_token: self._on_dqn_training_event(
                        event, dict(data, _workbench_run_token=token)),
                )
        except Exception as exc:
            self.log("FAIL", f"[TRAIN] config error: {exc}")
            self.lbl_training_progress.config(text=f"Config error: {exc}", fg=ui.COLOR_RED)
            return

        self._training_history = _new_training_history()
        if spec.key == "double_dqn":
            self._restore_training_trend(_source_trend_records(self), _source_evaluation_records(self))
        self._draw_training_chart()
        self._training_running = True
        self._static_test_running = True
        self._training_last_random_step = 0
        self._reset_training_metrics()
        self.lbl_training_status.config(text="STARTING", fg=_status_color("STARTING"))
        self.lbl_training_progress.config(
            text=f"{spec.label} · {self.var_training_level.get()} · starting...",
            fg=ui.COLOR_BLUE,
        )
        self._set_training_controls(True)
        self.log(
            "HEADER",
            f"\n========== GYM TRAINING {spec.label.upper()} "
            f"{self.var_training_level.get()} =========="
        )

        if spec.key == "random_legal":
            self._training_pause_event = threading.Event()
            self._training_pause_event.set()
            self._training_stop_event = threading.Event()
            config = RandomAgentConfig(
                episodes=int(params["episodes"]),
                max_steps=int(params["max_steps"]),
                random_seed=seed,
            )
            threading.Thread(
                target=self._random_training_worker,
                args=(level_id, config, ticks, playthrough, seed_types,
                      self._training_max_episode_ticks),
                daemon=True,
            ).start()
            return

        if spec.key == "double_dqn":
            self._set_training_checkpoint_controls(True)
            threading.Thread(target=self._dqn_training_worker, daemon=True).start()
            return

        self._finish_training(spec.key, RuntimeError("trainer backend is not wired"))

    def _random_training_worker(self, level_id, config, ticks, playthrough, seed_types,
                                max_episode_ticks):
        env = instance = None
        try:
            from types import SimpleNamespace
            from train.runtime import TrainingRuntimeConfig
            from train.sampler import native_environment
            environment = EnvConfig(level_id=level_id, playthrough=playthrough,
                ticks_per_step=ticks, seed_selection_mode=self._training_seed_selection_mode,
                max_episode_ticks=max_episode_ticks)
            settings = SimpleNamespace(env=environment, runtime=TrainingRuntimeConfig(), reward_scheme="transition_dense")
            env, instance = native_environment(settings, settings.runtime, 0,
                getattr(self.client, "dll_path", None), seed_types)

            def on_event(event: str, data: dict):
                if event == "seed_selection":
                    self.root.after(0, self._apply_dqn_training_event, event, data)
                elif event == "step":
                    self._training_last_random_step = int(data["step"])
                    if int(data["step"]) % 10 == 0:
                        self.root.after(0, self._random_step_ui, data)
                elif event == "episode_end":
                    self.root.after(0, self._random_episode_ui, data)
                elif event not in ("episode_start", "finished"):
                    detail = data.get("error") or data.get("action_result") or event
                    self.root.after(
                        0,
                        self.log,
                        "FAIL",
                        f"[TRAIN:{event}] episode={data.get('episode')} "
                        f"step={data.get('step', '-')} {detail}",
                    )

            result = run_random_agent(
                env,
                config,
                pause_event=self._training_pause_event,
                stop_event=self._training_stop_event,
                on_event=on_event,
            )
        except Exception as exc:
            result = exc
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            if instance is not None:
                instance.close()
        self.root.after(0, self._finish_training, "random_legal", result)

    def _dqn_training_worker(self):
        controller = self._training_controller
        try:
            _seed_history_file(controller.run_dir / "episodes.jsonl", _source_trend_records(self))
            _seed_history_file(controller.run_dir / "evaluations.jsonl", _source_evaluation_records(self))
            result = controller.run()
        except Exception as exc:
            result = exc
        self.root.after(0, self._finish_training, "double_dqn", result)

    def _on_dqn_training_event(self, event: str, data: dict):
        if event in ("visualization", "visualization_clear"):
            bridge = getattr(self, "_training_visualization_bridge", None)
            if bridge is not None:
                bridge.post(data.get("_workbench_run_token"), event, data)
            return
        # Controller callbacks run on the worker thread. Tk must only be touched via root.after.
        self.root.after(0, self._apply_dqn_training_event, event, data)

    def _apply_dqn_training_event(self, event: str, data: dict):
        if event == "seed_selection":
            from pvz_gym.adventure_curriculum import SPECIAL_LEVELS
            special = data.get("special")
            level_id = data.get("level_id")
            if not special and level_id in SPECIAL_LEVELS:
                special = SPECIAL_LEVELS[level_id]
            if not special and hasattr(self, "var_training_level"):
                try:
                    lvl_str = self.var_training_level.get()
                    area, sub = map(int, lvl_str.split("-"))
                    cur_lvl = (area - 1) * 10 + sub
                    if cur_lvl in SPECIAL_LEVELS:
                        special = SPECIAL_LEVELS[cur_lvl]
                except Exception:
                    pass

            if special or data.get("mode") == "special":
                spec_name = special or "特殊关卡"
                self.log("INFO", f"[TRAIN:SEED] 本关为特殊关卡【{spec_name}】，由传送带/特殊规则供牌，无需选卡")
                return

            from pvz_gym.default_decks import PLANT_NAMES
            names = [PLANT_NAMES[i] if 0 <= i < 40 else f"植物{i}" for i in data.get("actual", [])]
            text = "本局选卡：" + ("、".join(names) or "无交互选卡")
            warnings = data.get("warnings", [])
            if warnings:
                text += "\n" + "；".join(warnings)
            if not self.training_deck_panel.busy:
                self.training_deck_panel.show_result(text)
            self.log("INFO", f"[TRAIN:SEED] {text}")
            return
        if event == "replay_discount_converted":
            self.log("INFO", f"[TRAIN:DQN] replay Gamma={data['gamma']:.10g} / "
                     f"{data['reference_ticks']} ticks: recalculated {data['recomputed_labels']} labels; "
                     f"cleared {data['invalidated_labels']} labels without event timing; TD data retained")
            return
        if event == "state":
            state = str(data.get("state", "RUNNING"))
            self.lbl_training_status.config(text=state, fg=_status_color(state))
            paused = state == "PAUSED" and bool(data.get("acknowledged", False))
            self._set_training_controls(self._training_running, paused)
            if state in ("PAUSING", "RESUMING", "RESETTING", "STOPPING", "ERROR"):
                self.btn_training_pause.set_disabled(True)
                self.btn_training_continue.set_disabled(True)
                if hasattr(self, "btn_training_reset_game"):
                    self.btn_training_reset_game.set_disabled(True)
            if data.get("error"):
                self.log("FAIL", f"[TRAIN:CONTROL] {data['error']}")
            return
        if event == "step":
            summary = data.get("summary", {})
            self._update_dqn_metrics(summary)
            self.lbl_training_progress.config(
                text=(
                    f"E{data.get('episode', 0)} · step {data.get('env_step', 0)} · "
                    f"clock {data.get('game_clock', 0)} · "
                    f"ticks {data.get('ticks_advanced', '-')}/{data.get('ticks_requested', '-')} · "
                    f"{data.get('result', 'UNKNOWN')}"
                ),
                fg=ui.COLOR_BLUE,
            )
            if int(data.get("mowers_used", 0) or 0) > 0:
                self.log("WARN", _format_dqn_mower_log(data))
            return
        if event == "episode_end":
            summary = data.get("summary", {})
            self._update_dqn_metrics(summary)
            loss = data.get("loss")
            curr_ep = int(data.get("episode", 0))
            # 防御性自愈：如果当前对局重新从第 1 局起步，而历史走势中残留有更大的对局编号（说明开启了新训练/微调），
            # 立即清空不相关的旧 Run 历史，防止旧对局与新对局重叠绘制造成折线混乱
            hist_episodes = self._training_history.get("episode", [])
            if curr_ep == 1 and hist_episodes and max(hist_episodes) > 1:
                self._training_history = _new_training_history()
            self._training_history["episode"].append(curr_ep)
            self._record_training_outcome(str(data.get("result", "UNKNOWN")))
            self._training_history["win_rate"].append(float(data.get("win_rate", 0.0)))
            self._record_training_reward(float(data.get("training_episode_reward", 0.0)))
            self._record_training_loss(float(loss) if loss is not None else None)
            self._draw_training_chart()
            result = str(data.get("result", "UNKNOWN"))
            display_result = "TRUNCATED" if data.get("truncated") else result
            self.log(
                "PASS" if result == "WON" else "WARN",
                f"[TRAIN:DQN] episode={data.get('episode')} {display_result} "
                f"steps={data.get('episode_steps')} "
                f"ticks={data.get('episode_ticks', 0)}/{data.get('max_episode_ticks')} "
                f"reward={float(data.get('training_episode_reward', 0.0)):.3f} "
                f"mowers={data.get('mowers_used', 0)} "
                f"first_mower_tick={data.get('first_mower_tick')} "
                f"win_rate={float(data.get('win_rate', 0.0)):.1%}",
            )
            return
        if event == "evaluation_pending":
            self.lbl_training_progress.config(text="等待各沙箱完成当前训练局，再并行评估", fg=ui.COLOR_BLUE)
            return
        if event == "level_sampling":
            rows = data.get("levels", [])
            lines = []
            for row in rows:
                level = int(row['level_id'])
                rate = row.get('recent_win_rate')
                rate_text = f"{rate:.0%}" if rate is not None else "—"
                lines.append(f"{(level-1)//10+1}-{(level-1)%10+1}/{row['playthrough']}周目 "
                    f"近{row['recent_episodes']}局胜率{rate_text} · 抽样{row['probability']:.1%} · "
                    f"局数占比{row['episode_share']:.1%}/步数占比{row['step_share']:.1%}")
            label = getattr(self, 'lbl_training_level_sampling', None)
            if label is not None:
                label.config(text=('自适应抽关' if data.get('enabled') else '均匀抽关') + '\n' + '\n'.join(lines))
            return
        if event == "training_cadence":
            self.log("INFO", f"[TRAIN] 实际更新间隔={data['effective_train_every']}步 · "
                     f"待更新={data['pending_learning_updates']}")
            return
        if event in ("evaluation_start", "evaluation_episode"):
            self.lbl_training_progress.config(
                text=f"并行评估：{data.get('parallel_envs', 1)} 个沙箱 · "
                     f"已完成 {data.get('completed', 0)}/{data.get('requested_episodes', 0)}",
                fg=ui.COLOR_BLUE)
            self._set_training_controls(self._training_running, False)
            return
        if event == "evaluation_end":
            self._record_training_evaluation(data)
            self._draw_training_chart()
            self.lbl_training_progress.config(
                text=f"评估{'完成' if data.get('complete') else '中断'}："
                     f"{data.get('episodes', 0)}/{data.get('requested_episodes', 0)} · "
                     f"胜率 {float(data.get('win_rate', 0)):.1%}", fg=ui.COLOR_BLUE)
            self._set_training_controls(self._training_running, False)
            for row in data.get('levels', []):
                level = int(row['level_id'])
                self.log("INFO", f"[TRAIN:EVAL] {(level-1)//10+1}-{(level-1)%10+1} "
                    f"{row['playthrough']}周目 · {row['wins']}/{row['episodes']}胜 · "
                    f"胜率{row['win_rate']:.1%}")
            return
        if event in ("evaluation_retry", "evaluation_error"):
            self.log("WARN", f"[TRAIN:EVAL] 沙箱 {data.get('environment_id', '-')} · "
                     f"seed={data.get('seed')} · {data.get('error', '')}")
            return
        if event == "worker_recovery":
            self.log("WARN", f"[TRAIN:RECOVERY] environment {data.get('environment_id')} · "
                     f"attempt={data.get('attempt')} / limit={data.get('restart_limit')}\n"
                     f"{data.get('error', '')}\n"
                     f"记录：{data.get('log_path', '')}")
            return
        if event in ("checkpoint", "checkpoint_loaded"):
            path = str(data.get("path", ""))
            if path:
                self._training_checkpoint_path = path
                suffix = " (loaded; replay empty)" if event == "checkpoint_loaded" else ""
                self.lbl_training_checkpoint.config(text=path + suffix)
                checkpoint_summary = data.get("checkpoint_summary")
                if event == "checkpoint_loaded" and isinstance(checkpoint_summary, dict):
                    self._show_checkpoint_metrics(checkpoint_summary, data)
            if event == "checkpoint_loaded" and self._training_controller is not None:
                # Repeated starts may have a newer checkpoint than the cached preview.
                # Reload the actual resumed run rather than dropping the intervening episodes.
                run_dir = self._training_controller.run_dir
                episode = int(data.get("episode", 0))
                step = int(data.get("env_step", 0))
                self._training_checkpoint_records = _load_trend_records(run_dir / "episodes.jsonl", episode)
                self._training_checkpoint_evaluations = _load_evaluation_records(
                    run_dir / "evaluations.jsonl", episode, step,
                )
                self._restore_training_trend(self._training_checkpoint_records, self._training_checkpoint_evaluations)
                self._draw_training_chart()
            return
        if event == "error":
            self.log("FAIL", f"[TRAIN:DQN] {data.get('error', 'unknown error')}")
        if event in ("transition_discarded", "visualization_unavailable", "episode_aborted"):
            self.log("WARN", f"[TRAIN:{event}] {data.get('error') or data.get('observation_quality') or data.get('reason', '')}")

    def _random_step_ui(self, data: dict):
        summary = data["summary"]
        self.lbl_training_status.config(text="RUNNING", fg=ui.COLOR_BLUE)
        self.lbl_training_progress.config(
            text=(
                f"E{data.get('episode', 0)}/{summary.requested_episodes} · "
                f"S{data.get('step', 0)} · clock {data.get('game_clock', 0)}"
            ),
            fg=ui.COLOR_BLUE,
        )
        self._update_random_metrics(summary, int(data.get("step", 0)))

    def _random_episode_ui(self, data: dict):
        summary = data["summary"]
        reward = float(data.get("reward", 0.0))
        self._update_random_metrics(summary, int(data.get("steps", 0)))
        self._training_history["episode"].append(int(data.get("episode", 0)))
        self._record_training_outcome(str(data.get("result", "UNKNOWN")))
        self._training_history["win_rate"].append(summary.win_rate)
        self._record_training_reward(reward)
        self._record_training_loss(None)
        self._draw_training_chart()
        result = str(data.get("result", "UNKNOWN"))
        self.log(
            "PASS" if result in ("WON", "LOST") else "WARN",
            f"[TRAIN:RANDOM] episode={data.get('episode')} {result} "
            f"steps={data.get('steps')} reward={reward:.1f}",
        )

    def _update_random_metrics(self, summary: RandomAgentSummary, step: int = 0):
        reward = summary.mean_reward if summary.episode_rewards else 0.0
        values = {
            "step": str(step),
            "episode": f"{summary.completed_episodes}/{summary.requested_episodes}",
            "record": f"{summary.wins} / {summary.losses}",
            "win_rate": f"{summary.win_rate:.1%}",
            "reward": f"{reward:.2f}",
            "loss": "-",
            "epsilon": "-",
            "replay": "-",
            "mowers": "-",
            "invalid": str(summary.invalid_actions),
            "ticks": "-",
            "long_ticks": "-",
        }
        self._set_training_metric_values(values)

    def _update_dqn_metrics(self, summary: dict):
        loss = summary.get("last_loss")
        values = {
            "step": f"{int(summary.get('env_steps', 0))}/{int(summary.get('requested_steps', 0))}",
            "episode": str(int(summary.get("episodes", 0))),
            "record": (
                f"{int(summary.get('wins', 0))} / {int(summary.get('losses', 0))}"
            ),
            "win_rate": f"{float(summary.get('win_rate', 0.0)):.1%}",
            "reward": f"{float(summary.get('mean_training_reward', 0.0)):.3f}",
            "loss": "-" if loss is None else f"{float(loss):.5f}",
            "epsilon": f"{float(summary.get('epsilon', 0.0)):.3f}",
            "replay": str(int(summary.get("replay_size", 0))),
            "mowers": str(int(summary.get("mowers_used", 0))),
            "invalid": str(int(summary.get("invalid_actions", 0))),
            "ticks": (
                f"{float(summary.get('mean_ticks_advanced', 0.0)):.1f} ± "
                f"{float(summary.get('std_ticks_advanced', 0.0)):.1f}"
            ),
            "long_ticks": (
                f"{int(summary.get('long_tick_jumps', 0))} "
                f"({float(summary.get('long_tick_jump_rate', 0.0)):.1%})"
            ),
        }
        self._set_training_metric_values(values)

    def _set_training_metric_values(self, values: dict[str, str]):
        for key, value in values.items():
            label = self._training_metric_labels.get(key)
            if label is not None:
                label.config(text=value)

    def _reset_training_metrics(self):
        self._set_training_metric_values(
            {
                "step": "0",
                "episode": "0",
                "record": "0 / 0",
                "win_rate": "0.0%",
                "reward": "0.000",
                "loss": "-",
                "epsilon": "-",
                "replay": "-",
                "mowers": "0",
                "invalid": "0",
                "ticks": "-",
                "long_ticks": "-",
            }
        )

    def _record_training_outcome(self, result: str):
        outcome = 1 if result == "WON" else 0 if result == "LOST" else None
        self._training_history["outcome"].append(outcome)
        recent = [value for value in self._training_history["outcome"][-20:] if value is not None]
        self._training_history["recent_win_rate"].append(
            sum(recent) / len(recent) if recent else None
        )

    def _record_training_reward(self, reward: float):
        history = self._training_history
        previous_count = len(history["reward"])
        previous_mean = history["mean_reward"][-1] if previous_count else 0.0
        history["reward"].append(reward)
        history["mean_reward"].append(
            previous_mean + (reward - previous_mean) / (previous_count + 1)
        )
        recent = history["reward"][-20:]
        history["recent_reward"].append(sum(recent) / len(recent))

    def _record_training_loss(self, loss: float | None):
        history = self._training_history
        previous_count = history["loss_samples"][-1] if history["loss_samples"] else 0
        previous_mean = history["mean_loss"][-1] if previous_count else 0.0
        history["loss"].append(loss)
        count = previous_count + int(loss is not None)
        history["loss_samples"].append(count)
        history["mean_loss"].append(
            previous_mean + (loss - previous_mean) / count
            if loss is not None else previous_mean if count else None
        )
        recent = [value for value in history["loss"][-5:] if value is not None]
        history["recent_loss"].append(sum(recent) / len(recent) if recent else None)

    def _record_training_evaluation(self, record: dict):
        if not record.get("complete") or int(record.get("episodes", 0)) <= 0:
            return
        # A callback replay must not add a second point for the same evaluation.
        evaluations = self._training_history["evaluation"]
        identity = (record["episode"], record["env_step"], record.get("kind", "full"))
        evaluations[:] = [item for item in evaluations
                          if (item["episode"], item["env_step"], item.get("kind", "full")) != identity]
        evaluations.append(dict(record))
        evaluations.sort(key=lambda item: (int(item["episode"]), int(item["env_step"])))

    def _restore_training_trend(self, records: list[dict], evaluations: list[dict] | None = None):
        self._training_history = _new_training_history()
        for record in records:
            self._training_history["episode"].append(int(record["episode"]))
            self._record_training_outcome(str(record.get("result", "UNKNOWN")))
            self._training_history["win_rate"].append(float(record.get("win_rate", 0.0)))
            self._record_training_reward(float(record.get("training_episode_reward", 0.0)))
            loss = record.get("loss")
            self._record_training_loss(float(loss) if loss is not None else None)
        for record in evaluations or []:
            self._record_training_evaluation(record)

    def _on_trend_preset_changed(self):
        preset = self.var_trend_preset.get()
        episodes = self._training_history.get("episode", [])
        last_ep = episodes[-1] if episodes else 1
        first_ep = episodes[0] if episodes else 1
        if preset == "全部 (E1-latest)":
            self.var_trend_start.set(str(first_ep))
            self.var_trend_end.set("latest")
        elif preset == "最近 100 局":
            self.var_trend_start.set(str(max(first_ep, last_ep - 100 + 1)))
            self.var_trend_end.set("latest")
        elif preset == "最近 200 局":
            self.var_trend_start.set(str(max(first_ep, last_ep - 200 + 1)))
            self.var_trend_end.set("latest")
        elif preset == "最近 400 局":
            self.var_trend_start.set(str(max(first_ep, last_ep - 400 + 1)))
            self.var_trend_end.set("latest")
        elif preset == "最近 800 局":
            self.var_trend_start.set(str(max(first_ep, last_ep - 800 + 1)))
            self.var_trend_end.set("latest")
        self._draw_training_chart()

    def _apply_trend_custom_range(self):
        if hasattr(self, "var_trend_preset"):
            self.var_trend_preset.set("自定义")
        self._draw_training_chart()

    def _reset_trend_range(self):
        episodes = self._training_history.get("episode", [])
        first_ep = episodes[0] if episodes else 1
        if hasattr(self, "var_trend_preset"):
            self.var_trend_preset.set("全部 (E1-latest)")
        if hasattr(self, "var_trend_start"):
            self.var_trend_start.set(str(first_ep))
        if hasattr(self, "var_trend_end"):
            self.var_trend_end.set("latest")
        self._draw_training_chart()

    def _draw_training_chart(self):
        canvas = self.training_chart
        canvas.delete("all")
        width = max(canvas.winfo_width(), 420)
        height = max(canvas.winfo_height(), 318)
        raw_episodes = self._training_history.get("episode", [])
        evaluations = self._training_history.get("evaluation", [])

        if not raw_episodes:
            canvas.create_text(
                width / 2, height / 2,
                text="Waiting for completed episodes",
                fill=ui.CLR_TEXT_MUTED, font=("Microsoft YaHei UI", 9),
            )
            return

        min_avail_ep = raw_episodes[0]
        max_avail_ep = max([raw_episodes[-1]] + [int(record["episode"]) for record in evaluations])

        preset = getattr(self, "var_trend_preset", None)
        preset_val = preset.get() if preset else "全部 (E1-latest)"
        is_latest_pinned = False

        if preset_val == "最近 100 局":
            target_start = max(min_avail_ep, max_avail_ep - 100 + 1)
            target_end = max_avail_ep
            is_latest_pinned = True
        elif preset_val == "最近 200 局":
            target_start = max(min_avail_ep, max_avail_ep - 200 + 1)
            target_end = max_avail_ep
            is_latest_pinned = True
        elif preset_val == "最近 400 局":
            target_start = max(min_avail_ep, max_avail_ep - 400 + 1)
            target_end = max_avail_ep
            is_latest_pinned = True
        elif preset_val == "最近 800 局":
            target_start = max(min_avail_ep, max_avail_ep - 800 + 1)
            target_end = max_avail_ep
            is_latest_pinned = True
        elif preset_val == "全部 (E1-latest)":
            target_start = min_avail_ep
            target_end = max_avail_ep
            is_latest_pinned = True
        else:
            start_str = self.var_trend_start.get().strip().lstrip("eE") if hasattr(self, "var_trend_start") else ""
            end_str = self.var_trend_end.get().strip().lower() if hasattr(self, "var_trend_end") else "latest"
            try:
                target_start = int(start_str) if start_str else min_avail_ep
            except ValueError:
                target_start = min_avail_ep

            if "latest" in end_str or not end_str:
                target_end = max_avail_ep
                is_latest_pinned = True
            else:
                try:
                    target_end = int(end_str.lstrip("eE"))
                    is_latest_pinned = False
                except ValueError:
                    target_end = max_avail_ep
                    is_latest_pinned = True

        if target_start > target_end:
            target_start, target_end = target_end, target_start

        # 筛选选定范围内的对局记录索引
        valid_indices = [i for i, ep in enumerate(raw_episodes) if target_start <= ep <= target_end]
        row_h = height / 3.0
        x0, x1 = 83.0, float(width - 18)

        if not valid_indices:
            for row in range(3):
                y_top = row * row_h
                canvas.create_text(
                    x0, y_top + row_h / 2,
                    text=f"选定区间 [E{target_start} ~ E{target_end}] 内暂无对局记录",
                    fill=ui.CLR_TEXT_MUTED, font=("Microsoft YaHei UI", 8), anchor="w",
                )
            return

        view_episodes = [raw_episodes[i] for i in valid_indices]
        missing_ranges = [(left, right) for left, right in zip(view_episodes, view_episodes[1:])
                          if right - left > 1]
        first_episode = view_episodes[0]
        last_episode = view_episodes[-1]
        evaluations = [record for record in evaluations if target_start <= int(record["episode"]) <= target_end]

        def episode_x(ep):
            return (x1 if first_episode == last_episode else
                    x0 + (x1 - x0) * (ep - first_episode) / (last_episode - first_episode))

        series = (
            ("recent_win_rate", "Win rate · train / greedy", ui.COLOR_GREEN),
            ("recent_reward", "Mean reward · last 20 / all", ui.COLOR_BLUE),
            ("recent_loss", "Mean TD loss ×100 · last 5 / all", ui.COLOR_RED),
        )

        for row, (key, title, color) in enumerate(series):
            display_scale = 100.0 if key == "recent_loss" else 1.0
            raw_vals = self._training_history[key]
            values = [
                raw_vals[i] * display_scale if raw_vals[i] is not None else None
                for i in valid_indices
            ]
            y_top = row * row_h
            y0, y1 = y_top + 39.0, (row + 1) * row_h - 25.0

            if row:
                canvas.create_line(0, y_top, width, y_top, fill=ui.CLR_BORDER)
            if width < 600:
                title = {"recent_win_rate": "Win rate", "recent_reward": "Mean reward",
                         "recent_loss": "Mean TD loss ×100"}[key]
            title_item = canvas.create_text(
                8, y_top + 15,
                text=title,
                fill=ui.CLR_TEXT_MAIN, font=("Microsoft YaHei UI", 8, "bold"),
                anchor="w", tags="trend_title",
            )
            canvas.create_text(
                x0, y_top + 29,
                text=("Train: last 20 / dashed all · orange dots: greedy"
                      if key == "recent_win_rate" else "Solid: last 20  ·  dashed: all episodes"
                      if key == "recent_reward"
                      else "Episode-end samples · solid: last 5 · dashed: all"),
                fill=ui.CLR_TEXT_MUTED, font=("Microsoft YaHei UI", 7), anchor="w",
            )
            available = [val for val in values if val is not None]
            if not available and not (key == "recent_win_rate" and evaluations and view_episodes):
                canvas.create_text(
                    x0, (y0 + y1) / 2,
                    text=("Waiting for completed episodes" if not view_episodes
                          else "No completed win/loss result" if key == "recent_win_rate"
                          else "No optimizer updates recorded"),
                    fill=ui.CLR_TEXT_MUTED, font=("Microsoft YaHei UI", 8), anchor="w",
                )
                continue

            if key == "recent_win_rate":
                low, high = 0.0, 1.0
                latest = f"{available[-1]:.1%}" if available else "-"
                all_wr = self._training_history['win_rate'][valid_indices[-1]] if valid_indices else 0.0
                detail = f"all {all_wr:.1%}"
                fmt = lambda val: f"{val:.0%}"
            else:
                reference_key = "mean_reward" if key == "recent_reward" else "mean_loss"
                raw_ref = self._training_history[reference_key]
                ref_vals = [
                    raw_ref[i] * display_scale for i in valid_indices
                    if raw_ref[i] is not None
                ]
                scale_values = available + ref_vals
                low = min(0.0, min(scale_values)) if scale_values else 0.0
                high = max(0.0, max(scale_values)) if scale_values else 1.0
                span = high - low
                pad = max(span * 0.12, abs(high) * 0.08, 0.01)
                low -= pad
                high += pad
                latest = f"{available[-1]:.3f}"
                all_val = raw_ref[valid_indices[-1]] * display_scale if valid_indices and raw_ref[valid_indices[-1]] is not None else 0.0
                detail = f"all {all_val:.3f}"
                fmt = lambda val: f"{val:.2f}"

            latest_item = canvas.create_text(
                width - 8, y_top + 15,
                text=(f"Last {latest}  ·  {detail}  ·  n={len(available)}" if width >= 600
                      else f"{latest}  ·  {detail}"),
                fill=color, font=("Consolas", 8), anchor="e",
                tags="trend_latest",
            )
            if key == "recent_win_rate" and evaluations:
                canvas.itemconfigure(latest_item, text=f"Last {latest}  ·  {detail}")
                latest_eval = evaluations[-1]
                greedy_item = canvas.create_text(
                    canvas.bbox(latest_item)[0] - 12, y_top + 15,
                    text=f"Greedy {float(latest_eval['win_rate']):.1%}",
                    fill=GREEDY_TREND_COLOR, font=("Consolas", 8), anchor="e",
                    tags="greedy_evaluation",
                )
                if canvas.bbox(greedy_item)[0] < canvas.bbox(title_item)[2] + 8:
                    canvas.itemconfigure(latest_item, text=f"Last {latest}")
                    canvas.coords(greedy_item, canvas.bbox(latest_item)[0] - 12, y_top + 15)

            for fraction in (0.0, 0.5, 1.0):
                y = y1 - (y1 - y0) * fraction
                tick = low + (high - low) * fraction
                canvas.create_line(x0, y, x1, y, fill=ui.CLR_BORDER)
                canvas.create_text(
                    x0 - 7, y, text=fmt(tick), fill=ui.CLR_TEXT_MUTED,
                    font=("Consolas", 7), anchor="e",
                )

            if key != "recent_win_rate" and low < 0 < high:
                zero_y = y1 - (y1 - y0) * (-low) / (high - low)
                canvas.create_line(x0, zero_y, x1, zero_y, fill=ui.CLR_TEXT_MUTED, dash=(3, 3))

            # 绘制全量虚线均线
            ref_key = "win_rate" if key == "recent_win_rate" else "mean_reward" if key == "recent_reward" else "mean_loss"
            raw_ref_all = self._training_history[ref_key]
            reference_values = [
                raw_ref_all[i] * display_scale if raw_ref_all[i] is not None else None
                for i in valid_indices
            ]
            for indices in _trend_segments(reference_values, int(x1 - x0)):
                reference = []
                last_ep = None
                for index in indices:
                    curr_ep = view_episodes[index]
                    if last_ep is not None and curr_ep - last_ep > 30 and len(reference) >= 4:
                        canvas.create_line(*reference, fill=ui.CLR_TEXT_MUTED, width=1, dash=(3, 3))
                        reference = []
                    last_ep = curr_ep
                    norm = (reference_values[index] - low) / (high - low)
                    reference.extend((episode_x(curr_ep), y1 - (y1 - y0) * norm))
                if len(reference) >= 4:
                    canvas.create_line(*reference, fill=ui.CLR_TEXT_MUTED, width=1, dash=(3, 3))

            # 绘制近期实线走势
            for indices in _trend_segments(values, int(x1 - x0)):
                points = []
                last_ep = None
                for index in indices:
                    curr_ep = view_episodes[index]
                    if last_ep is not None and curr_ep - last_ep > 30 and len(points) >= 4:
                        canvas.create_line(*points, fill=color, width=2, smooth=False)
                        points = []
                    last_ep = curr_ep
                    norm = (values[index] - low) / (high - low)
                    points.extend((episode_x(curr_ep), y1 - (y1 - y0) * norm))
                if len(points) >= 4:
                    canvas.create_line(*points, fill=color, width=2, smooth=False)
                elif points:
                    canvas.create_oval(
                        points[0] - 2, points[1] - 2,
                        points[0] + 2, points[1] + 2,
                        fill=color, outline="",
                    )

            if available:
                last_index = max(index for index, val in enumerate(values) if val is not None)
                last_x = episode_x(view_episodes[last_index])
                last_y = y1 - (y1 - y0) * (available[-1] - low) / (high - low)
                canvas.create_oval(
                    last_x - 3, last_y - 3, last_x + 3, last_y + 3,
                    fill=color, outline=ui.CLR_PANEL,
                )

            if key == "recent_win_rate" and evaluations:
                for record in evaluations:
                    rec_ep = int(record["episode"])
                    if first_episode <= rec_ep <= last_episode:
                        x = episode_x(rec_ep)
                        y = y1 - (y1 - y0) * float(record["win_rate"])
                        canvas.create_oval(x - 3, y - 3, x + 3, y + 3,
                                           fill=GREEDY_TREND_COLOR, outline=ui.CLR_PANEL,
                                           tags="greedy_evaluation")

            for left, right in missing_ranges:
                gap_left, gap_right = episode_x(left), episode_x(right)
                if gap_right - gap_left >= 100:
                    canvas.create_text(
                        (gap_left + gap_right) / 2, (y0 + y1) / 2,
                        text=f"缺失 {right - left - 1} 局记录",
                        fill=ui.CLR_TEXT_MUTED, font=("Microsoft YaHei UI", 7),
                        tags="trend_gap",
                    )

            # X 轴刻度标注
            canvas.create_text(
                x0, y1 + 12, text=f"E{first_episode}",
                fill=ui.CLR_TEXT_MUTED, font=("Consolas", 7), anchor="w",
            )
            end_label = f"E{last_episode}" + (" (latest)" if is_latest_pinned else "")
            canvas.create_text(
                x1, y1 + 12, text=end_label,
                fill=ui.CLR_TEXT_MUTED, font=("Consolas", 7),
                anchor="e",
            )
            if last_episode - first_episode > 120:
                mid_ep = (first_episode + last_episode) // 2
                canvas.create_text(
                    episode_x(mid_ep), y1 + 12, text=f"E{mid_ep}",
                    fill=ui.CLR_TEXT_MUTED, font=("Consolas", 7), anchor="center",
                )

    def _load_training_checkpoint(self):
        if self._training_running or self._selected_training_spec().key != "double_dqn":
            return
        path = filedialog.askopenfilename(
            title="Load Double DQN checkpoint",
            filetypes=(("PyTorch checkpoint", "*.pt"), ("All files", "*.*")),
        )
        if path:
            self._training_checkpoint_path = path
            self._training_checkpoint_config = None
            self._training_loaded_config = None
            self._training_checkpoint_loading = True
            self.lbl_training_checkpoint.config(text=path + " (reading metadata...)")
            self._set_training_checkpoint_controls(True)
            threading.Thread(
                target=self._inspect_training_checkpoint_worker,
                args=(path,),
                daemon=True,
            ).start()

    def _inspect_training_checkpoint_worker(self, path: str):
        try:
            _agent, config, payload = DoubleDQNAgent.from_checkpoint(path, device="cpu", weights_only=True)
            records = _load_trend_records(
                Path(path).parent / "episodes.jsonl",
                int(payload.get("episode", 0) or 0),
            )
            payload = dict(payload)
            payload["seed_types"] = load_checkpoint_seed_types(path, config, payload)
            payload["trend_records"] = records
            payload["evaluation_records"] = _load_evaluation_records(
                Path(path).parent / "evaluations.jsonl",
                int(payload.get("episode", 0) or 0), int(payload.get("env_step", 0) or 0),
            )
            if not isinstance(payload.get("training_summary"), dict) or not payload.get("training_summary"):
                recovered = load_episode_summary(
                    Path(path).parent,
                    up_to_episode=int(payload.get("episode", 0) or 0),
                )
                if recovered:
                    payload = dict(payload)
                    payload["training_summary"] = recovered
                    payload["training_summary_source"] = "episodes.jsonl"
        except Exception as exc:
            self.root.after(0, self._checkpoint_inspection_failed, path, exc)
            return
        self.root.after(0, self._checkpoint_inspection_finished, path, config, payload)

    def _checkpoint_inspection_failed(self, path: str, exc: Exception):
        if self._training_checkpoint_path != path:
            return
        self._training_checkpoint_path = None
        self._training_checkpoint_config = None
        self._training_loaded_config = None
        self._training_checkpoint_loading = False
        self._training_checkpoint_records = []
        self._training_checkpoint_evaluations = []
        self.lbl_training_checkpoint.config(text=f"Cannot load {Path(path).name}: {exc}")
        self.lbl_training_progress.config(text=f"Checkpoint error: {exc}", fg=ui.COLOR_RED)
        self.log("FAIL", f"[TRAIN:DQN] checkpoint load failed: {exc}")
        self._set_training_controls(False)

    def _checkpoint_inspection_finished(self, path: str, config, payload: dict):
        if self._training_checkpoint_path != path:
            return
        self._training_checkpoint_config = config
        self._training_checkpoint_loading = False
        if config is not None:
            source_config = config
            seed_types = payload.get("seed_types")
            if getattr(self, "_training_resume_run_dir", None) is not None:
                source_config = self._training_resume_config or config
                seed_types = load_run_seed_types(self._training_resume_run_dir)
            self._training_loaded_config = source_config
            self._apply_training_config(source_config, seed_types)
        conversion = payload.get("gamma_conversion")
        if conversion:
            self.log("INFO", f"[TRAIN:DQN] Gamma converted: {conversion['source_gamma']} / "
                     f"{conversion['source_ticks']} ticks -> {conversion['gamma']:.10g} / 100 ticks; "
                     "edit Gamma to train with a new discount.")
        self._training_checkpoint_records = payload.get("trend_records", [])
        self._training_checkpoint_evaluations = payload.get("evaluation_records", [])
        self._restore_training_trend(self._training_checkpoint_records, self._training_checkpoint_evaluations)
        self._draw_training_chart()
        summary = payload.get("training_summary")
        summary = dict(summary) if isinstance(summary, dict) else {}
        fallback = {
            "env_steps": int(payload.get("env_step", 0)),
            "episodes": int(payload.get("episode", 0)),
        }
        self._show_checkpoint_metrics(summary, fallback)
        step = int(payload.get("env_step", 0))
        episode = int(payload.get("episode", 0))
        if summary:
            record = f"{int(summary.get('wins', 0))}W/{int(summary.get('losses', 0))}L"
            score = float(summary.get("mean_training_reward", 0.0))
            win_rate = float(summary.get("win_rate", 0.0))
            recovered = payload.get("training_summary_source") == "episodes.jsonl"
            source = "recovered history · " if recovered else ""
            detail = f"{source}{record} · win {win_rate:.1%} · mean score {score:.3f}"
        else:
            detail = "legacy checkpoint · no saved performance history"
        self.lbl_training_checkpoint.config(
            text=f"{Path(path).name} · step {step} · episode {episode} · {detail}"
        )
        self.lbl_training_progress.config(
            text="Checkpoint loaded. Click Test loaded model for greedy evaluation.",
            fg=ui.COLOR_GREEN,
        )
        self.log("PASS", f"[TRAIN:DQN] loaded {path} · {detail}")
        self._set_training_checkpoint_controls(True)

    def _show_checkpoint_metrics(self, summary: dict, fallback: dict | None = None):
        fallback = fallback or {}
        has_history = bool(summary)
        wins = int(summary.get("wins", 0))
        losses = int(summary.get("losses", 0))
        values = {
            "step": str(int(summary.get("env_steps", fallback.get("env_steps", 0)))),
            "episode": str(int(summary.get("episodes", fallback.get("episodes", 0)))),
            "record": f"{wins} / {losses}" if has_history else "- / -",
            "win_rate": f"{float(summary.get('win_rate', 0.0)):.1%}" if has_history else "-",
            "reward": (
                f"{float(summary.get('mean_training_reward', 0.0)):.3f}"
                if has_history else "-"
            ),
            "loss": (
                "-" if summary.get("last_loss") is None
                else f"{float(summary['last_loss']):.5f}"
            ),
            "epsilon": (
                f"{float(summary.get('epsilon', 0.0)):.3f}" if has_history else "-"
            ),
            "replay": "-",
            "mowers": str(int(summary.get("mowers_used", 0))) if has_history else "-",
            "invalid": str(int(summary.get("invalid_actions", 0))) if has_history else "-",
            "ticks": (
                f"{float(summary.get('mean_ticks_advanced', 0.0)):.1f} ± "
                f"{float(summary.get('std_ticks_advanced', 0.0)):.1f}"
                if has_history and int(summary.get("tick_samples", 0)) > 0 else "-"
            ),
            "long_ticks": (
                f"{int(summary.get('long_tick_jumps', 0))} "
                f"({float(summary.get('long_tick_jump_rate', 0.0)):.1%})"
                if has_history and int(summary.get("tick_samples", 0)) > 0 else "-"
            ),
        }
        self._set_training_metric_values(values)

    def _test_training_checkpoint(self):
        path = self._training_checkpoint_path
        if self._training_running or not path:
            return
        try:
            level_id, ticks, playthrough, _seed, _mower, _ = (
                self._parse_common_training_config()
            )
            episodes = int(self.var_training_eval_episodes.get())
            if episodes <= 0:
                raise ValueError("Test games must be positive")
            _, seed_types = self.training_deck_panel.settings()
            from .training_runtime import runtime_settings
            loaded = self._training_loaded_config
            game_runtime, evaluation_runtime, _ = runtime_settings(
                self, env=loaded.env if loaded is not None else None,
                runtime=loaded.runtime if loaded is not None else None)
            self._training_eval_runtime = replace(evaluation_runtime,
                num_envs=1, process_ids=evaluation_runtime.process_ids[:1],
                asynchronous=False, inference_batch_size=1)
            self._training_eval_game_runtime = game_runtime
        except Exception as exc:
            self.lbl_training_progress.config(text=f"Test config error: {exc}", fg=ui.COLOR_RED)
            return

        level_label = self.var_training_level.get()
        simulation_mode = _selected_mode(self)
        if simulation_mode == "continuous":
            try:
                from .q_heatmap_overlay import get_q_heatmap_overlay
                if hasattr(self, "var_training_q_overlay") and self.var_training_q_overlay.get() == "关闭":
                    self.var_training_q_overlay.set("Top 5 热力图")
                overlay_enabled = True
                if hasattr(self, "var_training_q_overlay"):
                    overlay_enabled = (str(self.var_training_q_overlay.get()) == "Top 5 热力图")
                get_q_heatmap_overlay().set_enabled(overlay_enabled)
            except Exception:
                pass
        self._training_running = True
        self._training_evaluating = True
        self._static_test_running = True
        self._training_controller = None
        self._training_pause_event = None
        self._training_stop_event = threading.Event()
        from uuid import uuid4
        from .training_visualization import TrainingVisualizationBridge
        from .dqn_headless import get_q_heatmap_overlay
        self._training_run_token = uuid4().hex
        self._training_eval_visualize = (simulation_mode == "continuous"
            and str(self.var_training_q_overlay.get()) == "Top 5 热力图")
        bridge = getattr(self, "_training_visualization_bridge", None)
        if bridge is None:
            bridge = TrainingVisualizationBridge(self.root, overlay=get_q_heatmap_overlay())
            self._training_visualization_bridge = bridge
        bridge.start(self._training_run_token)
        self._set_training_controls(True)
        self.btn_training_pause.set_disabled(True)
        self.btn_training_continue.set_disabled(True)
        self.lbl_training_status.config(text="TESTING", fg=ui.COLOR_BLUE)
        self.lbl_training_progress.config(
            text=f"Testing {Path(path).name} · {level_label} · {simulation_mode} · 0/{episodes} games...",
            fg=ui.COLOR_BLUE,
        )
        self.log("HEADER", f"\n========== DQN CHECKPOINT TEST {Path(path).name} · {level_label} ==========")
        threading.Thread(
            target=self._checkpoint_evaluation_worker,
            args=(path, level_id, ticks, playthrough, episodes, seed_types, simulation_mode),
            daemon=True,
        ).start()

    def _checkpoint_evaluation_worker(
        self, path: str, level_id: int, ticks: int, playthrough: int,
        episodes: int, seed_types, simulation_mode: str
    ):
        run_token = getattr(self, "_training_run_token", None)
        def on_event(event, data):
            if event in ("visualization", "visualization_clear"):
                bridge = getattr(self, "_training_visualization_bridge", None)
                if bridge is not None:
                    bridge.post(run_token, event, data)
                return
            self.root.after(0, self._apply_checkpoint_evaluation_event,
                            event, dict(data, _workbench_run_token=run_token))
        try:
            result = evaluate_checkpoint(
                path,
                level_id=level_id,
                ticks_per_step=ticks,
                playthrough=playthrough,
                max_episode_ticks=self._training_max_episode_ticks or 0,
                seed_count=episodes,
                device="auto",
                dll_path=getattr(self.client, "dll_path", None),
                seed_types=seed_types,
                simulation_mode=simulation_mode,
                seed_selection_mode=self._training_seed_selection_mode,
                on_event=on_event,
                stop_event=self._training_stop_event,
                training_runtime=getattr(self, "_training_eval_runtime", None),
                game_runtime=getattr(self, "_training_eval_game_runtime", None),
                visualize_q=lambda: bool(getattr(self, "_training_eval_visualize", False)),
            )
        except Exception as exc:
            result = exc
        self.root.after(0, self._finish_checkpoint_evaluation, result)

    def _apply_checkpoint_evaluation_event(self, event: str, data: dict):
        if (data.get("_workbench_run_token") is not None
                and data["_workbench_run_token"] != getattr(self, "_training_run_token", None)):
            return
        if event == "visualization_unavailable":
            self.log("WARN", f"[EVAL:VISUAL] {data.get('error', '')}")
            return
        if event != "episode_end":
            return
        episode = int(data.get("episode", 0))
        episodes = int(data.get("episodes", 0))
        wins = int(data.get("wins", 0))
        losses = int(data.get("losses", 0))
        self._set_training_metric_values(
            {
                "step": str(int(data.get("steps", 0))),
                "episode": f"{episode}/{episodes}",
                "record": f"{wins} / {losses}",
                "win_rate": f"{float(data.get('win_rate', 0.0)):.1%}",
                "reward": f"{float(data.get('mean_score', 0.0)):.3f}",
                "loss": "-",
                "epsilon": "0 (greedy)",
                "replay": "-",
                "ticks": (
                    f"{float(data.get('mean_ticks_advanced', 0.0)):.1f} ± "
                    f"{float(data.get('std_ticks_advanced', 0.0)):.1f}"
                ),
                "long_ticks": (
                    f"{int(data.get('long_tick_jumps', 0))} "
                    f"({float(data.get('long_tick_jump_rate', 0.0)):.1%})"
                ),
            }
        )
        result = str(data.get("result", "UNKNOWN"))
        self.lbl_training_progress.config(
            text=(
                f"Test {episode}/{episodes} · {result} · "
                f"score {float(data.get('score', 0.0)):.3f} · "
                f"win rate {float(data.get('win_rate', 0.0)):.1%}"
            ),
            fg=ui.COLOR_GREEN if result == "WON" else ui.COLOR_RED,
        )
        self.log(
            "PASS" if result == "WON" else "WARN",
            f"[EVAL:DQN] {episode}/{episodes} seed={data.get('seed')} {result} "
            f"steps={data.get('steps')} score={float(data.get('score', 0.0)):.3f} "
            f"ticks={data.get('episode_ticks', 0)} "
            f"first_mower_tick={data.get('first_mower_tick')} "
            f"win_rate={float(data.get('win_rate', 0.0)):.1%}",
        )

    def _finish_checkpoint_evaluation(self, result):
        bridge = getattr(self, "_training_visualization_bridge", None)
        if bridge is not None:
            bridge.close()
        self._training_run_token = None
        self._training_eval_visualize = False
        try:
            from .q_heatmap_overlay import get_q_heatmap_overlay
            get_q_heatmap_overlay().clear()
        except Exception:
            pass
        self._training_running = False
        self._training_evaluating = False
        self._static_test_running = False
        self._training_pause_event = None
        self._training_stop_event = None
        self._set_training_controls(False)
        if isinstance(result, Exception):
            self.lbl_training_status.config(text="FAILED", fg=ui.COLOR_RED)
            self.lbl_training_progress.config(text=f"Model test failed: {result}", fg=ui.COLOR_RED)
            self.log("FAIL", f"[EVAL:DQN] {result}")
            return

        summary: EvaluationSummary = result
        status = "STOPPED" if summary.stopped else "FINISHED"
        detail = (
            f"tested {summary.episodes}/{summary.requested_episodes} · "
            f"{summary.wins}W/{summary.losses}L · win rate {summary.win_rate:.1%} · "
            f"mean score {summary.mean_score:.3f} · ticks "
            f"{summary.mean_ticks_advanced:.1f}±{summary.std_ticks_advanced:.1f} · "
            f"long jumps {summary.long_tick_jump_rate:.1%}"
        )
        self.lbl_training_status.config(text=status, fg=_status_color(status))
        self.lbl_training_progress.config(text=detail, fg=_status_color(status))
        self.log("PASS" if not summary.stopped else "WARN", f"[EVAL:DQN] {detail}")
        self.refresh_status()

    def _clear_training_checkpoint(self):
        if self._training_running:
            return
        self._training_checkpoint_path = None
        self._training_checkpoint_config = None
        self._training_loaded_config = None
        self._training_checkpoint_loading = False
        self._training_checkpoint_records = []
        self._training_checkpoint_evaluations = []
        self._training_history = _new_training_history()
        self._draw_training_chart()
        self.lbl_training_checkpoint.config(text="New training (no checkpoint loaded)")
        self._reset_training_metrics()
        self._set_training_checkpoint_controls(
            self._selected_training_spec().key == "double_dqn"
        )

    def _save_training_checkpoint(self):
        controller = self._training_controller
        if controller is None or self._selected_training_spec().key != "double_dqn":
            self.log("WARN", "[TRAIN] no Double DQN trainer is available to save")
            return
        path = filedialog.asksaveasfilename(
            title="Save Double DQN checkpoint",
            defaultextension=".pt",
            filetypes=(("PyTorch checkpoint", "*.pt"),),
        )
        if not path:
            return

        def worker():
            try:
                saved = controller.save_checkpoint(path)
                self.root.after(0, self.lbl_training_checkpoint.config, {"text": str(saved)})
                self.root.after(0, self.log, "PASS", f"[TRAIN] checkpoint saved: {saved}")
            except Exception as exc:
                self.root.after(0, self.log, "FAIL", f"[TRAIN] checkpoint save failed: {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def _finish_training(self, trainer_key: str, result):
        self._training_running = False
        self._static_test_running = False
        self._training_pause_event = None
        self._training_stop_event = None
        self._set_training_controls(False)

        if isinstance(result, Exception):
            self.lbl_training_status.config(text="FAILED", fg=ui.COLOR_RED)
            self.lbl_training_progress.config(text=str(result), fg=ui.COLOR_RED)
            self.log("FAIL", f"[TRAIN] exception: {result}")
            self.refresh_status()
            return

        if trainer_key == "random_legal":
            self._update_random_metrics(result, self._training_last_random_step)
            if result.stopped:
                status = "STOPPED"
            elif result.integrity_failures:
                status = "FAILED"
            else:
                status = "FINISHED"
            detail = (
                f"{result.completed_episodes}/{result.requested_episodes} · "
                f"{result.wins}W/{result.losses}L · errors={result.integrity_failures}"
            )
        else:
            summary: DoubleDQNSummary = result
            self._update_dqn_metrics(summary.snapshot())
            status = "STOPPED" if summary.stopped else "FINISHED"
            detail = (
                f"step {summary.env_steps}/{summary.requested_steps} · episodes {summary.episodes} · "
                f"win rate {summary.win_rate:.1%} · mowers {summary.mowers_used}"
            )
            if self._training_controller is not None and self._training_controller.latest_checkpoint:
                self._training_checkpoint_path = str(self._training_controller.latest_checkpoint)
                self.lbl_training_checkpoint.config(text=self._training_checkpoint_path)

        self.lbl_training_status.config(text=status, fg=_status_color(status))
        self.lbl_training_progress.config(text=detail, fg=_status_color(status))
        self.log("PASS" if status == "FINISHED" else "WARN", f"[TRAIN] {status}: {detail}")
        if status == "STOPPED":
            self.log("PASS", "[TRAIN] 训练已安全停止，各环境已退出，最新模型与 Replay 均已完整持久化保存。")
        self.refresh_status()

    def _pause_training(self):
        if not self._training_running:
            return
        if self._training_controller is not None:
            self.log("INFO", "[TRAIN] 收到暂停请求，正在等待各游戏环境完成当前决策步后挂起...")
            self.lbl_training_progress.config(
                text="正在暂停：等待各环境当前决策步执行完毕...",
                fg=ui.COLOR_GOLD,
            )
            self._training_controller.pause()
            self.lbl_training_status.config(text=self._training_controller.state,
                                            fg=_status_color(self._training_controller.state))
            self._set_training_controls(True, False)
            self.btn_training_pause.set_disabled(True)
            return
        elif self._training_pause_event is not None:
            self.log("INFO", "[TRAIN] 收到暂停请求，正在挂起训练...")
            self._training_pause_event.clear()
        self.lbl_training_status.config(text="PAUSED", fg=_status_color("PAUSED"))
        self._set_training_controls(True, True)

    def _continue_training(self):
        if not self._training_running:
            return
        self.log("INFO", "[TRAIN] 正在继续训练，恢复环境采样与网络学习...")
        if self._training_controller is not None:
            if self._training_controller.state != "PAUSED":
                self.log("WARN", "[TRAIN:CONTROL] Continue 等待 Pause 确认后才可执行。")
                return
            new_mode = _selected_mode(self)
            if hasattr(self._training_controller, "set_simulation_mode"):
                try:
                    self._training_controller.set_simulation_mode(new_mode)
                    self.log("INFO", f"[TRAIN:DQN] 训练模式已同步为: {new_mode}")
                except Exception as exc:
                    self.log("FAIL", f"[TRAIN:DQN] 模式同步失败: {exc}")
                    return
            if new_mode == "continuous":
                try:
                    from .q_heatmap_overlay import get_q_heatmap_overlay
                    if hasattr(self, "var_training_q_overlay") and self.var_training_q_overlay.get() == "关闭":
                        self.var_training_q_overlay.set("Top 5 热力图")
                    overlay_enabled = True
                    if hasattr(self, "var_training_q_overlay"):
                        overlay_enabled = (str(self.var_training_q_overlay.get()) == "Top 5 热力图")
                    get_q_heatmap_overlay().set_enabled(overlay_enabled)
                except Exception:
                    pass
            # 动态热同步探索策略
            if hasattr(self._training_controller, "set_exploration_settings") and "exploration_mode" in self._training_param_vars:
                try:
                    modes = {label: key for key, label in EXPLORATION_MODE_LABELS.items()}
                    mode_label = self._training_param_vars["exploration_mode"].get()
                    mode_key = modes.get(mode_label, "epsilon_random")
                    ratio_str = self._training_param_vars["exploration_ratio"].get()
                    ratio_val = (
                        DEFAULT_EXPLORATION_RATIO
                        if mode_key == "epsilon_random"
                        else parse_exploration_ratio(ratio_str)
                    )
                    weights_str = self._training_param_vars["exploration_type_weights"].get()
                    weights_val = parse_action_family_weights(weights_str)
                    self._training_controller.set_exploration_settings(
                        mode=mode_key,
                        ratio=ratio_val,
                        type_weights=weights_val,
                    )
                    ratio_desc = f" ({format_exploration_ratio(ratio_val)})" if mode_key != "epsilon_random" else ""
                    self.log("INFO", f"[TRAIN:DQN] 探索策略已动态同步为: {mode_label}{ratio_desc}")
                except Exception as exp_err:
                    self.log("WARN", f"[TRAIN:DQN] 探索策略同步失败: {exp_err}")
                    return
            # 动态热同步回放机制
            if hasattr(self._training_controller, "set_replay_settings") and "replay_scheme" in self._training_param_vars:
                try:
                    scheme_val = str(self._training_param_vars["replay_scheme"].get())
                    alpha_val = float(self._training_param_vars["per_alpha"].get()) if "per_alpha" in self._training_param_vars else None
                    beta_val = float(self._training_param_vars["per_beta_start"].get()) if "per_beta_start" in self._training_param_vars else None
                    self._training_controller.set_replay_settings(
                        scheme=scheme_val,
                        alpha=alpha_val,
                        beta=beta_val,
                    )
                    self.log("INFO", f"[TRAIN:DQN] 回放机制已动态同步为: {scheme_val} (alpha={alpha_val}, beta_start={beta_val})")
                except Exception as replay_err:
                    self.log("WARN", f"[TRAIN:DQN] 回放机制同步失败: {replay_err}")
                    return
            self._training_controller.resume()
            self.lbl_training_status.config(text=self._training_controller.state,
                                            fg=_status_color(self._training_controller.state))
            self._set_training_controls(True, False)
            self.btn_training_pause.set_disabled(True)
            return
        elif self._training_pause_event is not None:
            self._training_pause_event.set()
        self.lbl_training_status.config(text="RUNNING", fg=ui.COLOR_BLUE)
        self._set_training_controls(True, False)

    def _reset_training_game(self):
        controller = self._training_controller
        if not self._training_running or controller is None:
            return
        try:
            controller.reset_game()
        except Exception as exc:
            self.log("FAIL", f"[TRAIN:RESET] {exc}")
            return
        self.lbl_training_status.config(text=controller.state, fg=_status_color(controller.state))
        self._set_training_controls(True, False)
        self.btn_training_pause.set_disabled(True)
        self.btn_training_continue.set_disabled(True)
        self.btn_training_reset_game.set_disabled(True)

    def _stop_training(self):
        try:
            from .q_heatmap_overlay import get_q_heatmap_overlay
            get_q_heatmap_overlay().clear()
        except Exception:
            pass
        if not self._training_running:
            return
        if self._training_controller is not None:
            self.log("WARN", "[TRAIN] 收到停止请求，正在等待各游戏环境同步退出并保存最新模型与 Replay...")
            self._training_controller.stop()
        elif self._training_stop_event is not None:
            self.log("WARN", "[TRAIN] 收到停止请求，正在等待训练线程结束...")
            self._training_stop_event.set()
            if self._training_pause_event is not None:
                self._training_pause_event.set()
        self.lbl_training_status.config(text="STOPPING", fg=_status_color("STOPPING"))
        self.lbl_training_progress.config(
            text="正在停止：等待多进程环境同步退出并持久化最新模型与 Replay...",
            fg=ui.COLOR_GOLD,
        )
        self._set_training_controls(True, False)

    def _build_left_panel_with_training(self, parent):
        original_build_left_panel(self, parent)
        _build_training_panel(self, parent)

    def _begin_live_guarded(self, title: str):
        if getattr(self, "_training_running", False):
            self.log("WARN", "[LIVE] Gym Training is running")
            return False
        return original_begin_live(self, title)

    gui_cls._selected_training_spec = _selected_training_spec
    gui_cls._training_trainer_changed = _training_trainer_changed
    gui_cls._set_training_checkpoint_controls = _set_training_checkpoint_controls
    gui_cls._set_training_controls = _set_training_controls
    gui_cls._refresh_training_exploration_controls = _refresh_training_exploration_controls
    gui_cls._refresh_training_replay_controls = _refresh_training_replay_controls
    gui_cls._refresh_training_auto_train_controls = _refresh_training_auto_train_controls
    gui_cls._refresh_training_evaluation_controls = _refresh_training_evaluation_controls
    gui_cls._parse_common_training_config = _parse_common_training_config
    gui_cls._parse_trainer_parameters = _parse_trainer_parameters
    gui_cls._start_training = _start_training
    gui_cls._on_mixed_level_settings_changed = _on_mixed_level_settings_changed
    gui_cls._random_training_worker = _random_training_worker
    gui_cls._dqn_training_worker = _dqn_training_worker
    gui_cls._on_dqn_training_event = _on_dqn_training_event
    gui_cls._apply_dqn_training_event = _apply_dqn_training_event
    gui_cls._random_step_ui = _random_step_ui
    gui_cls._random_episode_ui = _random_episode_ui
    gui_cls._update_random_metrics = _update_random_metrics
    gui_cls._update_dqn_metrics = _update_dqn_metrics
    gui_cls._set_training_metric_values = _set_training_metric_values
    gui_cls._reset_training_metrics = _reset_training_metrics
    gui_cls._record_training_outcome = _record_training_outcome
    gui_cls._record_training_reward = _record_training_reward
    gui_cls._record_training_loss = _record_training_loss
    gui_cls._restore_training_trend = _restore_training_trend
    gui_cls._record_training_evaluation = _record_training_evaluation
    gui_cls._on_trend_preset_changed = _on_trend_preset_changed
    gui_cls._apply_trend_custom_range = _apply_trend_custom_range
    gui_cls._reset_trend_range = _reset_trend_range
    gui_cls._draw_training_chart = _draw_training_chart
    gui_cls._load_training_checkpoint = _load_training_checkpoint
    gui_cls._inspect_training_checkpoint_worker = _inspect_training_checkpoint_worker
    gui_cls._checkpoint_inspection_failed = _checkpoint_inspection_failed
    gui_cls._checkpoint_inspection_finished = _checkpoint_inspection_finished
    gui_cls._show_checkpoint_metrics = _show_checkpoint_metrics
    gui_cls._test_training_checkpoint = _test_training_checkpoint
    gui_cls._checkpoint_evaluation_worker = _checkpoint_evaluation_worker
    gui_cls._apply_checkpoint_evaluation_event = _apply_checkpoint_evaluation_event
    gui_cls._finish_checkpoint_evaluation = _finish_checkpoint_evaluation
    gui_cls._clear_training_checkpoint = _clear_training_checkpoint
    gui_cls._save_training_checkpoint = _save_training_checkpoint
    gui_cls._finish_training = _finish_training
    gui_cls._pause_training = _pause_training
    gui_cls._reset_training_game = _reset_training_game
    gui_cls._continue_training = _continue_training
    gui_cls._stop_training = _stop_training
    gui_cls._build_left_panel = _build_left_panel_with_training
    gui_cls._begin_live = _begin_live_guarded
    gui_cls._training_gui_installed = True
