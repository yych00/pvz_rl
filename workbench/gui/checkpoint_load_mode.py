# -*- coding: utf-8 -*-
"""Workbench UI for explicit Double DQN checkpoint load semantics."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any

from train.dqn.optimized_controller import (
    CHECKPOINT_LOAD_RESUME,
    CHECKPOINT_LOAD_WEIGHTS_ONLY,
    CHECKPOINT_LOAD_WEIGHTS_REPLAY,
    CHECKPOINT_LOAD_STRICT,
    CHECKPOINT_LOAD_MODEL,
)


_NEW_MODEL_LABEL = "训练新模型 (从零开始)"
_TRANSFER_LABEL = "跨关卡微调 / 评估 (基于权重)"
_REPLAY_TRANSFER_LABEL = "权重 + Replay 微调 (新训练)"
_RESUME_LABEL = "接续历史训练 (恢复进度)"
_STRICT_LABEL = "完整接续 (模型与经验池配对)"
_MODEL_LABEL = "模型接续 (经验池重新积累)"
_MODE_LABELS = (_NEW_MODEL_LABEL, _TRANSFER_LABEL, _REPLAY_TRANSFER_LABEL,
                _STRICT_LABEL, _MODEL_LABEL, _RESUME_LABEL)


def _is_resume_label(label):
    return label in (_RESUME_LABEL, _STRICT_LABEL, _MODEL_LABEL)


def _mode_key(label: str) -> str:
    if label == _REPLAY_TRANSFER_LABEL:
        return CHECKPOINT_LOAD_WEIGHTS_REPLAY
    if label == _STRICT_LABEL:
        return CHECKPOINT_LOAD_STRICT
    if label == _MODEL_LABEL:
        return CHECKPOINT_LOAD_MODEL
    if "Transfer" in label or "微调" in label or label == "Initialize from weights":
        return CHECKPOINT_LOAD_WEIGHTS_ONLY
    return CHECKPOINT_LOAD_RESUME


def install_checkpoint_load_mode_gui(gui_cls: Any, ui: Any) -> None:
    if getattr(gui_cls, "_checkpoint_load_mode_gui_installed", False):
        return

    original_build_left_panel = gui_cls._build_left_panel
    original_set_checkpoint_controls = gui_cls._set_training_checkpoint_controls
    original_load_checkpoint = gui_cls._load_training_checkpoint
    original_clear_checkpoint = gui_cls._clear_training_checkpoint
    original_dqn_worker = gui_cls._dqn_training_worker
    original_apply_dqn_event = gui_cls._apply_dqn_training_event
    original_resume_run = getattr(gui_cls, "_resume_training_run", None)

    def _checkpoint_mode_changed(self, _event=None):
        if not hasattr(self, "var_training_checkpoint_load_mode"):
            return
        if getattr(self, "_training_resume_resolving", False):
            return
        label = self.var_training_checkpoint_load_mode.get()
        if label in (_TRANSFER_LABEL, _REPLAY_TRANSFER_LABEL):
            # A weights transfer starts a fresh run even after selecting Resume Run.
            self._training_resume_run_dir = None
            self._training_resume_config = None
            self._training_resume_selected_mode = None
            self._training_resume_resolution_error = None
        elif (_is_resume_label(label)
              and getattr(self, "_training_resume_run_dir", None) is not None
              and not getattr(self, "_training_running", False)
              and getattr(self, "_training_resume_selected_mode", None) != _mode_key(label)):
            self._reselect_training_resume_checkpoint(_mode_key(label))
            return
        include_source_history = (
            _is_resume_label(label)
            or getattr(self, "_training_resume_run_dir", None) is not None
        )
        previous = getattr(self, "_training_include_source_history", None)
        previous_mode = getattr(self, "_training_trend_preview_mode", None)
        self._training_include_source_history = include_source_history
        self._training_trend_preview_mode = label
        if (_is_resume_label(label) and previous_mode != label
                and getattr(self, "_training_loaded_config", None) is not None
                and not getattr(self, "_training_running", False)):
            self._apply_training_config(self._training_loaded_config,
                                        getattr(self, "_training_loaded_seed_types", None))
        if (
            previous_mode is not None
            and (previous_mode != label or previous != include_source_history)
            and not getattr(self, "_training_running", False)
        ):
            # Preview source performance in either checkpoint mode. Only resume
            # carries these records into the next training run.
            self._restore_training_trend(
                self._training_checkpoint_records if label != _NEW_MODEL_LABEL else [],
                getattr(self, "_training_checkpoint_evaluations", []) if label != _NEW_MODEL_LABEL else [],
            )
            self._draw_training_chart()
        if label == _NEW_MODEL_LABEL:
            if not getattr(self, "_loading_training_checkpoint", False):
                self._training_checkpoint_path = None
                self._training_resume_run_dir = None
                self._training_resume_config = None
                self._training_resume_selected_mode = None
                self._training_resume_resolution_error = None
                self.lbl_training_checkpoint.config(text="未加载模型 (从零全新训练)")
            note = "★ 从零全新训练：网络权重随机初始化，步数从0开始，不依赖任何检查点。"
            color = ui.CLR_TEXT_MUTED
        elif label == _REPLAY_TRANSFER_LABEL:
            note = (
                "★ 权重 + Replay 微调：加载模型及同一保存点的经验池，创建独立新 Run；"
                "步数、优化器和探索调度重新开始。可调整关卡和训练参数；"
                "经验池缺失、配对失败或奖励语义不兼容时停止加载。"
                "请选择完整保存的模型（如 stopped.pt / final.pt），模型自动保存可能不带经验池。"
            )
            color = ui.COLOR_BLUE if getattr(self, "_training_checkpoint_path", None) else ui.COLOR_GOLD
        elif label == _TRANSFER_LABEL:
            if not getattr(self, "_training_checkpoint_path", None):
                note = "请点击下方【加载权重 (.pt)】选择预训练模型；随后可在上方自由切换关卡进行测试或微调。"
                color = ui.COLOR_GOLD
            else:
                note = (
                    "★ 跨关卡微调/评估：仅继承模型权重，步数重置为0。可在上方【关卡】中切换目标关卡，"
                    "直接点击 Smoke/Test 评估或【开始训练】微调。"
                )
                color = ui.COLOR_BLUE
        else:
            if not getattr(self, "_training_checkpoint_path", None):
                note = "请点击下方【恢复训练 Run】或【加载权重 (.pt)】选择历史训练记录。"
                color = ui.COLOR_GOLD
            else:
                note = (
                    "接续训练：已恢复历史配置，锁定网络结构、关卡、周目、种子和卡组。"
                    "开始前可调整学习率、探索、训练频率、容量、Gamma、n-step及奖励；"
                    "修改奖励会重建经验池。暂停时可调整探索参数。"
                )
                color = ui.COLOR_GREEN
                if label == _STRICT_LABEL:
                    note = "完整接续：必须具有相同保存点的模型、优化器、RNG和经验池；缺失或不匹配会停止加载。游戏回合重新开始。"
                elif label == _MODEL_LABEL:
                    note = "模型接续：保留模型、优化器和训练进度，经验池明确清空并重新 warmup。游戏回合重新开始。"
                else:
                    note += " 兼容接续在经验池不匹配时会明确报告并重建；严格配对请选择完整接续。"
            error = getattr(self, "_training_resume_resolution_error", None)
            if error:
                note = "无法按当前接续方式恢复此 Run：" + str(error)
                color = ui.COLOR_RED

        if hasattr(self, "lbl_training_checkpoint_mode_note"):
            self.lbl_training_checkpoint_mode_note.config(text=note, fg=color)
        if hasattr(self, "_refresh_training_reward_scheme"):
            self._refresh_training_reward_scheme()
        self._refresh_training_resume_controls()

    def _refresh_training_resume_controls(self):
        running = getattr(self, "_training_running", False)
        loading = getattr(self, "_training_checkpoint_loading", False)
        resolution_error = getattr(self, "_training_resume_resolution_error", None)
        is_dqn = self._selected_training_spec().key == "double_dqn"
        locked = (is_dqn and _is_resume_label(self.var_training_checkpoint_load_mode.get())
                  and bool(getattr(self, "_training_checkpoint_path", None)))
        for name, editable_state in (
            ("cmb_training_level", "readonly"), ("cmb_training_playthrough", "readonly"),
            ("ent_training_seed", "normal"), ("ent_training_seed_types", "normal"),
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.config(state="disabled" if running or locked or loading else editable_state)
        if hasattr(self, "training_deck_panel"):
            self.training_deck_panel.set_busy(running or locked or loading)
        if hasattr(self, "mixed_level_panel"):
            self.mixed_level_panel.set_busy(running or locked or loading)
        ticks = getattr(self, "ent_training_ticks", None)
        if ticks is not None:
            ticks.config(state="disabled" if running or loading else "normal")

        if loading:
            for widget in self._training_param_widgets.values():
                widget.config(state="disabled")
            for name in ("cmb_training_checkpoint_load_mode", "cmb_training_reward_scheme",
                         "ent_training_max_episode_ticks"):
                widget = getattr(self, name, None)
                if widget is not None:
                    widget.config(state="disabled")
            self.btn_training_run.set_disabled(True)
        elif resolution_error and is_dqn:
            self.btn_training_run.set_disabled(True)
        elif not running:
            self.btn_training_run.set_disabled(not self._selected_training_spec().available)

    def _build_left_panel(self, parent):
        self.var_training_checkpoint_load_mode = tk.StringVar(value=_NEW_MODEL_LABEL)
        original_build_left_panel(self, parent)

        actions = self.btn_training_load_checkpoint.master
        checkpoint_body = actions.master

        mode_panel = tk.Frame(
            checkpoint_body,
            bg=ui.CLR_PANEL,
            highlightbackground=ui.CLR_BORDER,
            highlightthickness=1,
        )
        mode_panel.pack(fill=tk.X, pady=(7, 0), before=actions)

        mode_header = tk.Frame(mode_panel, bg=ui.CLR_PANEL)
        mode_header.pack(fill=tk.X, padx=9, pady=(6, 0))
        tk.Label(
            mode_header,
            text="训练模式",
            font=("Microsoft YaHei UI", 9, "bold"),
            fg=ui.CLR_TEXT_MAIN,
            bg=ui.CLR_PANEL,
        ).pack(side=tk.LEFT)
        self.cmb_training_checkpoint_load_mode = ttk.Combobox(
            mode_header,
            textvariable=self.var_training_checkpoint_load_mode,
            values=_MODE_LABELS,
            state="readonly",
            width=28,
        )
        self.cmb_training_checkpoint_load_mode.pack(side=tk.RIGHT, padx=(12, 0), ipady=2)
        self.cmb_training_checkpoint_load_mode.bind(
            "<<ComboboxSelected>>", self._checkpoint_mode_changed
        )

        self.lbl_training_checkpoint_mode_note = tk.Label(
            mode_panel,
            text="",
            font=("Microsoft YaHei UI", 8),
            fg=ui.CLR_TEXT_MUTED,
            bg=ui.CLR_PANEL,
            anchor="w",
            justify=tk.LEFT,
            wraplength=720,
        )
        self.lbl_training_checkpoint_mode_note.pack(fill=tk.X, padx=9, pady=(5, 8))
        self._checkpoint_mode_changed()

        # 统一步骤按钮为一行 4 列等宽平衡布局
        resume_button = getattr(self, "btn_training_resume_run", None)
        for button in (
            resume_button,
            self.btn_training_load_checkpoint,
            self.btn_training_new_checkpoint,
            self.btn_training_save_checkpoint,
        ):
            if button is not None:
                button.grid_forget()

        # 本地化与美化按钮文案
        self.btn_training_load_checkpoint.set_text("加载权重 (.pt)")
        if resume_button is not None:
            resume_button.set_text("恢复训练 Run")
            resume_button.set_colors(ui.CLR_CARD, ui.COLOR_GREEN)
        self.btn_training_save_checkpoint.set_text("保存当前权重")
        self.btn_training_new_checkpoint.set_text("卸载已选模型")
        self.btn_training_new_checkpoint.set_colors(ui.CLR_CARD, ui.CLR_TEXT_MUTED)

        for col in range(4):
            actions.columnconfigure(col, weight=1, uniform="checkpoint-file-action")

        self.btn_training_load_checkpoint.grid(row=0, column=0, padx=(0, 2), sticky="ew")
        if resume_button is not None:
            resume_button.grid(row=0, column=1, padx=2, sticky="ew")
        self.btn_training_save_checkpoint.grid(row=0, column=2, padx=2, sticky="ew")
        self.btn_training_new_checkpoint.grid(row=0, column=3, padx=(2, 0), sticky="ew")

        try:
            is_dqn = self._selected_training_spec().key == "double_dqn"
        except Exception:
            is_dqn = False
        self._set_training_checkpoint_controls(is_dqn)

    def _set_training_checkpoint_controls(self, dqn: bool):
        original_set_checkpoint_controls(self, dqn)
        combo = getattr(self, "cmb_training_checkpoint_load_mode", None)
        if combo is not None:
            can_choose = dqn and not self._training_running
            combo.config(state="readonly" if can_choose else "disabled")
        if hasattr(self, "_checkpoint_mode_changed"):
            self._checkpoint_mode_changed()

    def _load_training_checkpoint(self):
        # The base loader refreshes controls before it returns. Keep the newly
        # selected path until this wrapper switches out of New model mode.
        self._loading_training_checkpoint = True
        try:
            result = original_load_checkpoint(self)
        finally:
            self._loading_training_checkpoint = False
        if getattr(self, "_training_checkpoint_path", None):
            if self.var_training_checkpoint_load_mode.get() == _NEW_MODEL_LABEL:
                self.var_training_checkpoint_load_mode.set(_TRANSFER_LABEL)
            self._set_training_checkpoint_controls(True)
            self._checkpoint_mode_changed()
        return result

    def _clear_training_checkpoint(self):
        result = original_clear_checkpoint(self)
        variable = getattr(self, "var_training_checkpoint_load_mode", None)
        if variable is not None:
            variable.set(_NEW_MODEL_LABEL)
        if hasattr(self, "_checkpoint_mode_changed"):
            self._checkpoint_mode_changed()
        return result

    def _resume_training_run(self):
        previous_label = self.var_training_checkpoint_load_mode.get()
        result = original_resume_run(self) if original_resume_run is not None else None
        if result and getattr(self, "_training_resume_run_dir", None) is not None:
            selected_label = previous_label if (_is_resume_label(previous_label)
                                                or previous_label == _REPLAY_TRANSFER_LABEL) else _RESUME_LABEL
            self.var_training_checkpoint_load_mode.set(selected_label)
            self._set_training_checkpoint_controls(True)
        if hasattr(self, "_checkpoint_mode_changed"):
            self._checkpoint_mode_changed()
        return result

    def _dqn_training_worker(self):
        controller = self._training_controller
        if controller is not None:
            mode = _mode_key(self.var_training_checkpoint_load_mode.get())
            controller.checkpoint_load_mode = mode
        return original_dqn_worker(self)

    def _apply_dqn_training_event(self, event: str, data: dict):
        original_apply_dqn_event(self, event, data)
        if event in ("checkpoint_restore_contract", "replay_restore_skipped"):
            if event == "replay_restore_skipped":
                prefix = "经验池重新积累：" if data.get("warmup_restarted", True) else "经验池加载失败："
                text = prefix + str(data.get("reason", ""))
            else:
                text = f"保存内容：{data.get('checkpoint_kind')} · 加载方式：{data.get('load_mode')} · 游戏回合重新开始"
                warnings = data.get("restoration_warnings", [])
                if isinstance(warnings, (list, tuple)) and warnings:
                    warnings = [str(message) for message in warnings]
                    text += " · 恢复提示：" + warnings[0]
                    if len(warnings) > 1:
                        text += f"（另 {len(warnings) - 1} 项，见日志）"
                    self.log("WARN", "[TRAIN:DQN] " + "; ".join(warnings))
            self.lbl_training_progress.config(text=text, fg=ui.COLOR_GOLD)
        if event != "checkpoint_loaded":
            return
        mode = _mode_key(self.var_training_checkpoint_load_mode.get())
        if mode not in (CHECKPOINT_LOAD_WEIGHTS_ONLY, CHECKPOINT_LOAD_WEIGHTS_REPLAY):
            return

        path = str(data.get("path", ""))
        self._reset_training_metrics()
        epsilon_var = self._training_param_vars.get("epsilon_start")
        try:
            epsilon_text = f"{float(epsilon_var.get()):.3f}" if epsilon_var is not None else "-"
        except (TypeError, ValueError):
            epsilon_text = "-"
        replay_size = int(data.get("replay_size", 0) or 0) if mode == CHECKPOINT_LOAD_WEIGHTS_REPLAY else 0
        replay_note = f"继承 Replay {replay_size} 条" if mode == CHECKPOINT_LOAD_WEIGHTS_REPLAY else "独立新经验池"
        self._set_training_metric_values({"epsilon": epsilon_text, "replay": str(replay_size)})
        self.lbl_training_checkpoint.config(
            text=(
                f"{Path(path).name} · 权重微调模式 · 步数重置为0 · {replay_note}"
                if path
                else f"已初始化权重 · 步数重置为0 · {replay_note}"
            )
        )
        self.lbl_training_progress.config(
            text="已将模型权重初始化到新训练会话中，可在上方关卡中自由训练。",
            fg=ui.COLOR_GREEN,
        )

    gui_cls._checkpoint_mode_changed = _checkpoint_mode_changed
    gui_cls._refresh_training_resume_controls = _refresh_training_resume_controls
    gui_cls._build_left_panel = _build_left_panel
    gui_cls._set_training_checkpoint_controls = _set_training_checkpoint_controls
    gui_cls._load_training_checkpoint = _load_training_checkpoint
    gui_cls._clear_training_checkpoint = _clear_training_checkpoint
    gui_cls._dqn_training_worker = _dqn_training_worker
    gui_cls._apply_dqn_training_event = _apply_dqn_training_event
    if original_resume_run is not None:
        gui_cls._resume_training_run = _resume_training_run
    gui_cls._checkpoint_load_mode_gui_installed = True


__all__ = ["install_checkpoint_load_mode_gui"]
