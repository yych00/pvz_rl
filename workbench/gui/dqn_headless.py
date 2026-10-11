# -*- coding: utf-8 -*-
"""Workbench selector for stepped, headless, and continuous DQN simulation."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from pvz_gym.env import set_default_simulation_mode as set_env_simulation_mode
from train.common import set_default_simulation_mode as set_train_simulation_mode
from .q_heatmap_overlay import get_q_heatmap_overlay


_MODE_OPTIONS = (
    "Turbo (最快，全程无头)",
    "Headless (Fast)",
    "Continuous (No Pause)",
    "Stepped",
)
_DEFAULT_MODE_LABEL = "Turbo (最快，全程无头)"

_Q_OVERLAY_OPTIONS = (
    "关闭",
    "Top 5 热力图",
)
_DEFAULT_Q_OVERLAY_LABEL = "关闭"


def _selected_mode(self) -> str:
    variable = getattr(self, "var_training_simulation_mode", None)
    value = str(variable.get()) if variable is not None else _DEFAULT_MODE_LABEL
    normalized = value.strip().lower()
    if normalized.startswith("turbo"):
        return "turbo"
    if normalized.startswith("headless"):
        return "headless"
    if normalized.startswith("continuous"):
        return "continuous"
    return "stepped"


def install_dqn_headless_gui(gui_cls: Any, ui: Any) -> None:
    if getattr(gui_cls, "_dqn_headless_gui_installed", False):
        return

    original_trainer_changed = gui_cls._training_trainer_changed
    original_set_controls = gui_cls._set_training_controls
    original_start_training = gui_cls._start_training

    def _training_trainer_changed(self):
        # training.py destroys every trainer-parameter child before rebuilding the
        # selected trainer. Clear cached widgets first so control-state updates do
        # not touch destroyed Tk objects.
        self.cmb_training_simulation = None
        self.cmb_training_q_overlay = None
        original_trainer_changed(self)
        try:
            spec = self._selected_training_spec()
        except Exception:
            return

        runtime_card = getattr(self, "training_runtime_card", None)
        checkpoint_card = getattr(self, "training_checkpoint_card", None)
        if spec.key != "double_dqn":
            if runtime_card is not None:
                runtime_card.pack_forget()
            return

        if runtime_card is not None and not runtime_card.winfo_ismapped():
            if checkpoint_card is not None:
                runtime_card.pack(fill=tk.X, padx=12, pady=(0, 8), before=checkpoint_card)
            else:
                runtime_card.pack(fill=tk.X, padx=12, pady=(0, 8))

        from .training_runtime import build_runtime_controls
        build_runtime_controls(self, ui)

    def _set_training_controls(self, running: bool, paused: bool = False):
        original_set_controls(self, running, paused)
        combobox = getattr(self, "cmb_training_simulation", None)
        try:
            spec = self._selected_training_spec()
            is_dqn = spec.key == "double_dqn"
            available = spec.available
        except Exception:
            is_dqn = False
            available = False
        if combobox is not None:
            try:
                controller = getattr(self, '_training_controller', None)
                from train.runtime import uses_parallel_runtime
                parallel = controller is not None and uses_parallel_runtime(controller.config.runtime)
                combobox_disabled = (running and (not paused or parallel)) or not is_dqn or not available
                combobox.config(
                    state="disabled" if combobox_disabled else "readonly"
                )
            except Exception:
                pass
        for runtime_widget, readonly in getattr(self, '_runtime_widgets', ()):
            try: runtime_widget.configure(state='disabled' if running else ('readonly' if readonly else 'normal'))
            except tk.TclError: pass

        # 同步 Q值显示 状态
        q_combobox = getattr(self, "cmb_training_q_overlay", None)
        if q_combobox is not None:
            try:
                is_continuous = _selected_mode(self) == "continuous"
                count_var = getattr(self, '_runtime_param_vars', {}).get('num_envs')
                single = count_var is None or str(count_var.get()) == '1'
                q_disabled = not is_dqn or not available or not is_continuous or not single
                q_combobox.config(
                    state="disabled" if q_disabled else "readonly"
                )
                if q_disabled:
                    if hasattr(self, "var_training_q_overlay"):
                        self.var_training_q_overlay.set(_DEFAULT_Q_OVERLAY_LABEL)
                    get_q_heatmap_overlay().set_visible(False)
                    get_q_heatmap_overlay().clear()
            except Exception:
                pass

        # A live trainer mutates model and replay independently between decision
        # boundaries. Active runs therefore use autosave/milestone checkpoints;
        # manual Save remains disabled until no trainer owns the controller.
        save_button = getattr(self, "btn_training_save_checkpoint", None)
        if save_button is not None and running:
            save_button.set_disabled(True)

    def _start_training(self):
        try:
            is_dqn = self._selected_training_spec().key == "double_dqn"
        except Exception:
            is_dqn = False
        mode = _selected_mode(self) if is_dqn else "stepped"
        # training.py constructs EnvConfig without a simulation_mode argument.
        # Set both defaults immediately before that construction so the saved
        # config and the PvZEnv created by the worker agree on the same mode.
        set_train_simulation_mode(mode)
        set_env_simulation_mode(mode)
        if is_dqn:
            self.log("INFO", f"[TRAIN:DQN] simulation={mode}")
            if mode == "continuous" and hasattr(self, "var_training_q_overlay"):
                if self.var_training_q_overlay.get() == "关闭":
                    self.var_training_q_overlay.set("Top 5 热力图")
                if str(self.var_training_q_overlay.get()) == "Top 5 热力图":
                    get_q_heatmap_overlay().set_enabled(True)
        return original_start_training(self)

    gui_cls._training_trainer_changed = _training_trainer_changed
    gui_cls._set_training_controls = _set_training_controls
    gui_cls._start_training = _start_training
    gui_cls._dqn_headless_gui_installed = True


__all__ = ["install_dqn_headless_gui"]
