# -*- coding: utf-8 -*-
"""Workbench reward-scheme selector for Double DQN."""

from __future__ import annotations

import tkinter as tk
from dataclasses import replace
from tkinter import ttk
from typing import Any

from train.dqn.config import DoubleDQNConfig
from train.dqn.reward import (
    REWARD_SCHEME_NAMES,
    get_reward_scheme,
)
from train.dqn.run_state import load_run_config

_REWARD_DESCRIPTIONS = {
    "sparse": "仅使用输赢奖励和小车扣分。",
    "balanced": "综合输赢、小车损失、威胁变化与终局表现。",
    "dense_early": "加入击杀、植物损失、波次和阳光变化。",
    "transition_dense": "逐步结算资产、有效伤害和威胁变化；整局净 shaping 平滑趋近 ±0.40，正负可抵消，同方向奖励逐渐变小而不硬截断。",
    "staged": "按原版价格统计阳光、植物槽、传送带、掉落卡和场上植物的价值变化。",
}


def install_reward_scheme_gui(gui_cls: Any, ui: Any) -> None:
    if getattr(gui_cls, "_reward_scheme_gui_installed", False):
        return

    original_build_left_panel = gui_cls._build_left_panel
    original_trainer_changed = gui_cls._training_trainer_changed
    original_set_training_controls = gui_cls._set_training_controls
    original_dqn_worker = gui_cls._dqn_training_worker
    original_checkpoint_finished = gui_cls._checkpoint_inspection_finished
    original_resume_run = getattr(gui_cls, "_resume_training_run", None)
    original_parse_common = gui_cls._parse_common_training_config

    def _set_badge(label, value: str, *, active: bool = True) -> None:
        label.config(
            text=value,
            fg=ui.CLR_TEXT_MAIN if active else ui.CLR_TEXT_MUTED,
        )

    def _refresh_training_reward_scheme(self, *, running: bool | None = None) -> None:
        combo = getattr(self, "cmb_training_reward_scheme", None)
        variable = getattr(self, "var_training_reward_scheme", None)
        note = getattr(self, "lbl_training_reward_note", None)
        if combo is None or variable is None or note is None:
            return

        try:
            dqn = self._selected_training_spec().key == "double_dqn"
        except Exception:
            dqn = False
        running = bool(getattr(self, "_training_running", False)) if running is None else running
        if not dqn:
            combo.config(state="disabled")
            note.config(text="奖励方案仅适用于 Double DQN。")
            return

        name = variable.get().strip() or "transition_dense"
        try:
            scheme = get_reward_scheme(name)
        except ValueError:
            name = "transition_dense"
            variable.set(name)
            scheme = get_reward_scheme(name)

        combo.config(state="disabled" if running else "readonly")
        # Sparse keeps the legacy manual mower field for old checkpoints. Named
        # presets are exact recipes and mirror their scalar into saved configs.
        sparse_penalty = None
        if scheme.name == "sparse":
            try:
                sparse_penalty = float(self.var_training_mower_penalty.get())
            except (TypeError, ValueError):
                sparse_penalty = abs(scheme.mower_used)
                self.var_training_mower_penalty.set(str(sparse_penalty))
            mower = -abs(float(sparse_penalty))
        else:
            self.var_training_mower_penalty.set(str(abs(scheme.mower_used)))
            mower = scheme.mower_used

        badge_labels = getattr(self, "_training_reward_badges", {})
        titles = getattr(self, "_training_reward_badge_titles", {})
        names = (("WIN / LOSS", "FIRST / LATER", "NET SHAPING", "ASSET", "DAMAGE")
                 if scheme.name == "transition_dense" else
                 (("WIN / LOSS", "MOWER", "STAGE CAP", "RESOURCE", "END DAMAGE")
                 if scheme.name == "staged" else
                 ("WIN / LOSS", "MOWER", "THREAT", "END PLANTS", "END SUN")))
        for key, title in zip(("terminal", "mower", "threat", "plants", "sun"), names):
            if key in titles:
                titles[key].config(text=title)
        if scheme.name == "transition_dense":
            if badge_labels:
                for key, value in (("terminal", "+1 / -1"), ("mower", "-0.50 / -0.10"),
                                   ("threat", "+0.40 / -0.40"), ("plants", "÷5000"), ("sun", "×0.30")):
                    _set_badge(badge_labels[key], value)
            note.config(text=(
                _REWARD_DESCRIPTIONS[scheme.name]
                + "\n普通种子槽不计资产；一次性卡牌和植物使用原始价格。"
                + "\n小车与终局独立结算；威胁权重 0.10，单步裁剪 ±0.01。"
                + "\n威胁比较同一批僵尸；排除出生、小车影响行和终局清场，伤害为保守快照估算。"
            ))
            return
        if scheme.name == "staged":
            if badge_labels:
                for key, value in (("terminal", "+1 / -1"), ("mower", "-0.20"),
                                   ("threat", "+0.10"), ("plants", "÷1000"), ("sun", "+0.50")):
                    _set_badge(badge_labels[key], value)
            note.config(text=(
                _REWARD_DESCRIPTIONS[scheme.name]
                + "\n每 1000 tick 结算：资源变化 /1000 + 0.5×掉血比例增量，正向最多 +0.1。"
                + "\n每辆小车单独扣 0.2；终局胜负 ±1，再加 0.5×最终掉血比例。"
            ))
            return
        if badge_labels:
            _set_badge(badge_labels["terminal"], "+1 / -1")
            _set_badge(badge_labels["mower"], f"{mower:+.2f}", active=mower != 0.0)
            _set_badge(
                badge_labels["threat"],
                f"±{abs(scheme.zombie_threat):.2f}",
                active=scheme.zombie_threat != 0.0,
            )
            _set_badge(
                badge_labels["plants"],
                f"+{scheme.terminal_plants:.2f}",
                active=scheme.terminal_plants != 0.0,
            )
            _set_badge(
                badge_labels["sun"],
                f"+{scheme.terminal_sun:.2f}",
                active=scheme.terminal_sun != 0.0,
            )

        note.config(
            text=(
                _REWARD_DESCRIPTIONS[scheme.name]
                + "\n"
                + "丢车在发生时计入该步奖励，后续丢车继续扣分。"
            )
        )

    def _build_left_panel(self, parent):
        original_build_left_panel(self, parent)
        self.var_training_reward_scheme = tk.StringVar(value="transition_dense")

        mower_entry = getattr(self, "ent_training_mower_penalty", None)
        if mower_entry is None:
            return
        mower_box = mower_entry.master
        reward_row = mower_box.master
        reward_body = reward_row.master

        # Rebuild the old single-line reward area into a compact dashboard.
        for child in reward_row.winfo_children():
            try:
                child.pack_forget()
            except Exception:
                pass

        # Match the equal-width field cards used by Trainer Parameters.
        for col in range(1):
            reward_row.columnconfigure(col, weight=1, uniform="reward-field")

        def field_card(column: int, title: str):
            card = tk.Frame(
                reward_row, bg=ui.CLR_PANEL,
                highlightbackground=ui.CLR_BORDER, highlightthickness=1,
            )
            card.grid(row=0, column=column, sticky="nsew",
                      padx=0)
            tk.Label(
                card, text=title, font=("Microsoft YaHei UI", 8, "bold"),
                fg=ui.CLR_TEXT_MUTED, bg=ui.CLR_PANEL, anchor="w",
            ).pack(fill=tk.X, padx=9, pady=(8, 0))
            return card

        self.cmb_training_reward_scheme = ttk.Combobox(
            field_card(0, "奖励方案"),
            textvariable=self.var_training_reward_scheme,
            values=REWARD_SCHEME_NAMES,
            state="readonly",
            width=12,
            font=ui.FONT_MAIN,
        )
        self.cmb_training_reward_scheme.pack(fill=tk.X, padx=9, pady=(4, 9), ipady=2)
        self.cmb_training_reward_scheme.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._refresh_training_reward_scheme(),
        )

        badges = tk.Frame(reward_body, bg=ui.CLR_CARD)
        badges.pack(fill=tk.X, pady=(9, 2))
        for col in range(5):
            badges.columnconfigure(col, weight=1, uniform="reward-badge")

        badge_specs = (
            ("terminal", "WIN / LOSS"),
            ("mower", "MOWER"),
            ("threat", "THREAT"),
            ("plants", "END PLANTS"),
            ("sun", "END SUN"),
        )
        self._training_reward_badges = {}
        self._training_reward_badge_titles = {}
        for index, (key, title) in enumerate(badge_specs):
            card = tk.Frame(
                badges,
                bg=ui.CLR_PANEL,
                highlightthickness=1,
                highlightbackground=ui.CLR_BORDER,
            )
            card.grid(
                row=0,
                column=index,
                sticky="nsew",
                padx=(0 if index == 0 else 3, 0 if index == 4 else 3),
            )
            title_label = tk.Label(
                card, text=title, font=("Microsoft YaHei UI", 7, "bold"),
                fg=ui.CLR_TEXT_MUTED, bg=ui.CLR_PANEL, anchor="center",
            )
            title_label.pack(fill=tk.X, padx=5, pady=(6, 1))
            self._training_reward_badge_titles[key] = title_label
            label = tk.Label(
                card,
                text="—",
                font=ui.FONT_MONO,
                fg=ui.CLR_TEXT_MAIN,
                bg=ui.CLR_PANEL,
                justify="center",
                padx=5,
                pady=0,
            )
            label.pack(fill=tk.X, pady=(0, 6))
            self._training_reward_badges[key] = label

        note = self.lbl_training_reward_note
        note.pack_forget()
        note.config(anchor="w", justify=tk.LEFT, wraplength=440)
        note.pack(fill=tk.X, pady=(7, 0))

        def wrap_reward_note(event):
            wraplength = max(120, event.width - 4)
            if int(note.cget("wraplength")) != wraplength:
                note.config(wraplength=wraplength)

        note.bind("<Configure>", wrap_reward_note, add="+")

        # Keep the legacy entry alive but hidden; resume/config code still reads it.
        mower_box.pack_forget()
        self._refresh_training_reward_scheme()

    def _training_trainer_changed(self):
        result = original_trainer_changed(self)
        self._refresh_training_reward_scheme()
        return result

    def _set_training_controls(self, running: bool, paused: bool = False):
        result = original_set_training_controls(self, running, paused)
        self._refresh_training_reward_scheme(running=running)
        return result

    def _dqn_training_worker(self):
        controller = self._training_controller
        if controller is not None:
            controller.config = replace(controller.config, **getattr(self, "_training_reward_options", {}))
        return original_dqn_worker(self)

    def _parse_common_training_config(self):
        result = original_parse_common(self)
        if self._selected_training_spec().key == "double_dqn":
            scheme = get_reward_scheme(self.var_training_reward_scheme.get().strip() or "transition_dense")
            updates = {
                "reward_scheme": scheme.name,
            }
            if scheme.name == "transition_dense":
                updates["shaping_limit_mode"] = "smooth_total"
            if scheme.name != "sparse":
                updates.update(reward_scale=scheme.terminal_scale, mower_penalty=abs(scheme.mower_used))
            # Validate and snapshot Tk values on the UI thread before dispatch.
            DoubleDQNConfig(**updates)
            self._training_reward_options = updates
        return result

    def _restore_reward_config(self, config):
        config = config or DoubleDQNConfig()
        self.var_training_reward_scheme.set(config.reward_scheme)
        self.var_training_mower_penalty.set(str(config.mower_penalty))
        self._refresh_training_reward_scheme()

    def _checkpoint_inspection_finished(self, path: str, config, payload: dict):
        if self._training_checkpoint_path != path:
            return
        result = original_checkpoint_finished(self, path, config, payload)
        variable = getattr(self, "var_training_reward_scheme", None)
        if variable is not None:
            self._restore_reward_config(getattr(self, "_training_loaded_config", None) or config)
        return result

    def _resume_training_run(self):
        if original_resume_run is None:
            return None
        result = original_resume_run(self)
        run_dir = getattr(self, "_training_resume_run_dir", None)
        variable = getattr(self, "var_training_reward_scheme", None)
        if run_dir is not None and variable is not None:
            try:
                config = load_run_config(run_dir)
                self._restore_reward_config(config)
            except Exception:
                pass
        return result

    gui_cls._refresh_training_reward_scheme = _refresh_training_reward_scheme
    gui_cls._restore_reward_config = _restore_reward_config
    gui_cls._parse_common_training_config = _parse_common_training_config
    gui_cls._training_trainer_changed = _training_trainer_changed
    gui_cls._set_training_controls = _set_training_controls
    gui_cls._dqn_training_worker = _dqn_training_worker
    gui_cls._checkpoint_inspection_finished = _checkpoint_inspection_finished
    if original_resume_run is not None:
        gui_cls._resume_training_run = _resume_training_run
    gui_cls._build_left_panel = _build_left_panel
    gui_cls._reward_scheme_gui_installed = True


__all__ = ["install_reward_scheme_gui"]
