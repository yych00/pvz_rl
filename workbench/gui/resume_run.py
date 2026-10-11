# -*- coding: utf-8 -*-
"""Workbench support for continuing a complete Double DQN run directory."""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog
from typing import Any

from train.dqn.run_state import load_run_config, load_run_seed_types, resolve_resume_checkpoint
from train.dqn.exploration import EXPLORATION_MODE_LABELS, format_exploration_ratio


def _level_label(level_id: int) -> str:
    value = int(level_id)
    if not 1 <= value <= 50:
        raise ValueError(f"invalid Adventure level_id in run config: {value}")
    return f"{(value - 1) // 10 + 1}-{(value - 1) % 10 + 1}"


def install_resume_run_gui(gui_cls: Any, ui: Any) -> None:
    if getattr(gui_cls, "_resume_run_gui_installed", False):
        return

    original_build_left_panel = gui_cls._build_left_panel
    original_set_checkpoint_controls = gui_cls._set_training_checkpoint_controls
    original_load_checkpoint = gui_cls._load_training_checkpoint
    original_clear_checkpoint = gui_cls._clear_training_checkpoint
    original_checkpoint_failed = gui_cls._checkpoint_inspection_failed
    original_start_training = gui_cls._start_training
    original_dqn_worker = gui_cls._dqn_training_worker

    def _build_left_panel(self, parent):
        original_build_left_panel(self, parent)
        self._training_resume_run_dir = None
        self._training_resume_config = None
        self._training_resume_selected_mode = None
        self._training_resume_resolution_error = None
        self._training_resume_resolving = False
        self._training_requested_autosave_every = 10_000
        load_button = getattr(self, "btn_training_load_checkpoint", None)
        if load_button is None:
            return
        actions = load_button.master
        self.btn_training_resume_run = ui.CyberButton(
            actions,
            "Resume run",
            self._resume_training_run,
            bg_color=ui.CLR_PANEL,
            fg_color=ui.COLOR_GREEN,
            height=32,
        )
        for col in range(4):
            actions.columnconfigure(col, weight=1, uniform="checkpoint-action")
        self.btn_training_resume_run.grid(row=0, column=0, padx=(0, 3), sticky="ew")
        self.btn_training_load_checkpoint.grid(row=0, column=1, padx=3, sticky="ew")
        self.btn_training_new_checkpoint.grid(row=0, column=2, padx=3, sticky="ew")
        self.btn_training_save_checkpoint.grid(row=0, column=3, padx=(3, 0), sticky="ew")
        try:
            is_dqn = self._selected_training_spec().key == "double_dqn"
        except Exception:
            is_dqn = False
        self._set_training_checkpoint_controls(is_dqn)

    def _set_training_checkpoint_controls(self, dqn: bool):
        original_set_checkpoint_controls(self, dqn)
        button = getattr(self, "btn_training_resume_run", None)
        if button is not None:
            button.set_disabled((not dqn) or self._training_running)

    def _apply_run_config(self, config, seed_types) -> None:
        self._training_loaded_seed_types = tuple(seed_types) if seed_types else None
        self.var_training_trainer.set("Double DQN")
        self._training_trainer_changed()

        self.var_training_level.set(_level_label(config.env.level_id))
        self.var_training_ticks.set(str(config.env.ticks_per_step))
        self.var_training_max_episode_ticks.set(str(config.env.max_episode_ticks or 0))
        self.var_training_playthrough.set(str(config.env.playthrough))
        self.var_training_seed.set(str(config.seed))
        seed_types = seed_types or config.env.seed_priority
        self.var_training_seed_types.set(
            "" if not seed_types else ",".join(str(value) for value in seed_types)
        )
        if hasattr(self, "training_deck_panel"):
            from .deck_panel import MODE_LABELS
            self.training_deck_panel.mode_var.set(MODE_LABELS[config.env.seed_selection_mode])
        if hasattr(self, "mixed_level_panel"):
            level_pool = getattr(config.env, "level_pool", None)
            self.mixed_level_panel.set_from_level_pool(
                level_pool,
                seed_selection_mode=config.env.seed_selection_mode,
            )
        self.var_training_mower_penalty.set(str(config.mower_penalty))

        parameter_values = {
            "total_steps": config.total_steps,
            "selected_action_forward": config.selected_action_forward,
            "learning_rate": config.learning_rate,
            "gamma": config.gamma,
            "batch_size": config.batch_size,
            "n_step": config.n_step,
            "replay_capacity": config.replay_capacity,
            "learning_starts": config.learning_starts,
            "train_every": config.train_every,
            "auto_train_every": getattr(config, "auto_train_every", True),
            "auto_train_high_watermark": getattr(config, "auto_train_high_watermark", 16),
            "auto_train_low_watermark": getattr(config, "auto_train_low_watermark", 4),
            "training_architecture": config.training_architecture,
            "policy_sync_interval": config.policy_sync_interval,
            "max_pending_updates": config.max_pending_updates,
            "gradient_steps": config.gradient_steps,
            "target_update_interval": config.target_update_interval,
            "epsilon_start": config.epsilon_start,
            "epsilon_end": config.epsilon_end,
            "epsilon_decay_steps": config.epsilon_decay_steps,
            "exploration_mode": EXPLORATION_MODE_LABELS[config.exploration_mode],
            "exploration_ratio": format_exploration_ratio(config.exploration_ratio),
            "exploration_type_weights": ",".join(
                str(value) for value in config.exploration_type_weights
            ),
            "autosave_every": config.autosave_every,
            "checkpoint_every": config.checkpoint_every,
            "device": config.device,
        }
        for key, value in parameter_values.items():
            variable = self._training_param_vars.get(key)
            if variable is not None:
                variable.set(str(value))
        for key in ("evaluation_mode", "evaluation_per_level", "evaluation_random_episodes",
                    "level_sampling_mode",
                    "adaptive_level_sampling", "level_sampling_window", "level_sampling_update_every",
                    "level_sampling_uniform_mix", "level_sampling_max_weight"):
            variable = self._training_param_vars.get(key)
            if variable is not None:
                variable.set(str(getattr(config, key)))

        simulation = getattr(self, "var_training_simulation_mode", None)
        if simulation is not None:
            mode_labels = {
                "turbo": "Turbo (最快，全程无头)",
                "headless": "Headless (Fast)",
                "continuous": "Continuous (No Pause)",
            }
            simulation.set(mode_labels.get(config.env.simulation_mode, "Stepped"))
        from .training_runtime import apply_runtime_settings
        apply_runtime_settings(self, config)

    def _select_training_resume_checkpoint(self, run_dir: Path, mode: str):
        # Keep the attempted Run and mode after a resolution error so choosing
        # another load mode can retry that Run without falling back silently.
        self._training_resume_run_dir = run_dir
        self._training_resume_selected_mode = mode
        self._training_resume_resolution_error = None
        self._training_resume_resolving = True
        self._training_checkpoint_records = []
        self._training_checkpoint_evaluations = []
        self._restore_training_trend([])
        self._reset_training_metrics()
        self._draw_training_chart()
        try:
            config = load_run_config(run_dir)
            seed_types = load_run_seed_types(run_dir)
            checkpoint = resolve_resume_checkpoint(run_dir, mode=mode)
            self._training_resume_config = config
            self._training_loaded_config = config
            self._training_checkpoint_path = str(checkpoint)
            self._training_checkpoint_config = config
            self._training_checkpoint_loading = True
            _apply_run_config(self, config, seed_types)
            if hasattr(self, "_restore_reward_config"):
                self._restore_reward_config(config)
        except Exception as exc:
            self._training_resume_resolution_error = str(exc)
            self._training_resume_config = None
            self._training_loaded_config = None
            self._training_checkpoint_path = None
            self._training_checkpoint_config = None
            self._training_checkpoint_loading = False
            self.lbl_training_checkpoint.config(text=f"恢复训练 Run 失败: {run_dir}")
            self.log("FAIL", f"[TRAIN] cannot resume run: {exc}")
            self.lbl_training_progress.config(
                text=f"Resume run error: {exc}",
                fg=ui.COLOR_RED,
            )
            return False
        finally:
            self._training_resume_resolving = False
            self._set_training_controls(False)

        self.lbl_training_checkpoint.config(
            text=f"恢复训练 Run: {run_dir} | {checkpoint.name} (读取性能记录...)"
        )
        # Show the performance snapshot embedded in newer checkpoints without
        # waiting for the resumed trainer thread to start.
        threading.Thread(
            target=self._inspect_training_checkpoint_worker,
            args=(str(checkpoint),),
            daemon=True,
        ).start()
        self.log(
            "PASS",
            f"[TRAIN:DQN] resume run={run_dir} checkpoint={checkpoint.name} "
            f"load_mode={mode} target_steps={config.total_steps}",
        )
        return True

    def _reselect_training_resume_checkpoint(self, mode: str):
        run_dir = getattr(self, "_training_resume_run_dir", None)
        if run_dir is None or self._training_running:
            return False
        return _select_training_resume_checkpoint(self, Path(run_dir), mode)

    def _resume_training_run(self):
        if self._training_running:
            return False
        directory = filedialog.askdirectory(
            title="Resume Double DQN run",
            initialdir="runs",
            mustexist=True,
        )
        if not directory:
            return False
        from .checkpoint_load_mode import _RESUME_LABEL, _REPLAY_TRANSFER_LABEL, _is_resume_label, _mode_key
        label = self.var_training_checkpoint_load_mode.get()
        if not _is_resume_label(label) and label != _REPLAY_TRANSFER_LABEL:
            self.var_training_checkpoint_load_mode.set(_RESUME_LABEL)
        return _select_training_resume_checkpoint(
            self, Path(directory), _mode_key(self.var_training_checkpoint_load_mode.get()),
        )

    def _load_training_checkpoint(self):
        previous_path = self._training_checkpoint_path
        previous_run = getattr(self, "_training_resume_run_dir", None)
        result = original_load_checkpoint(self)
        if self._training_checkpoint_path != previous_path:
            self._training_resume_run_dir = None
            self._training_resume_config = None
            self._training_resume_selected_mode = None
            self._training_resume_resolution_error = None
        else:
            self._training_resume_run_dir = previous_run
        return result

    def _clear_training_checkpoint(self):
        self._training_resume_run_dir = None
        self._training_resume_config = None
        self._training_resume_selected_mode = None
        self._training_resume_resolution_error = None
        return original_clear_checkpoint(self)

    def _checkpoint_inspection_failed(self, path: str, exc: Exception):
        belongs_to_run = (self._training_checkpoint_path == path
                          and getattr(self, "_training_resume_run_dir", None) is not None)
        if belongs_to_run:
            self._training_resume_resolution_error = str(exc)
        return original_checkpoint_failed(self, path, exc)

    def _start_training(self):
        try:
            is_dqn = self._selected_training_spec().key == "double_dqn"
        except Exception:
            is_dqn = False
        if is_dqn:
            error = getattr(self, "_training_resume_resolution_error", None)
            if error:
                self.log("FAIL", f"[TRAIN] cannot resume run: {error}")
                self.lbl_training_progress.config(text=f"Resume run error: {error}", fg=ui.COLOR_RED)
                self.btn_training_run.set_disabled(True)
                return
            variable = self._training_param_vars.get("autosave_every")
            if variable is not None:
                self._training_requested_autosave_every = int(variable.get())
        return original_start_training(self)

    def _dqn_training_worker(self):
        controller = self._training_controller
        run_dir = getattr(self, "_training_resume_run_dir", None)
        if controller is not None:
            if run_dir is not None:
                controller.run_dir = Path(run_dir)
            controller.config = replace(
                controller.config,
                autosave_every=int(getattr(self, "_training_requested_autosave_every", 10_000)),
            )
        return original_dqn_worker(self)

    gui_cls._resume_training_run = _resume_training_run
    gui_cls._reselect_training_resume_checkpoint = _reselect_training_resume_checkpoint
    gui_cls._apply_training_config = _apply_run_config
    gui_cls._load_training_checkpoint = _load_training_checkpoint
    gui_cls._clear_training_checkpoint = _clear_training_checkpoint
    gui_cls._checkpoint_inspection_failed = _checkpoint_inspection_failed
    gui_cls._start_training = _start_training
    gui_cls._set_training_checkpoint_controls = _set_training_checkpoint_controls
    gui_cls._dqn_training_worker = _dqn_training_worker
    gui_cls._build_left_panel = _build_left_panel
    gui_cls._resume_run_gui_installed = True


__all__ = ["install_resume_run_gui"]
