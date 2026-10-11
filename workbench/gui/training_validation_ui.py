# -*- coding: utf-8 -*-
"""Polished validation controls for model evaluation and smoke tests."""

from __future__ import annotations

import tkinter as tk
from typing import Any


def install_training_validation_ui(gui_cls: Any, ui: Any) -> None:
    if getattr(gui_cls, "_training_validation_ui_installed", False):
        return

    original_build_left_panel = gui_cls._build_left_panel
    original_set_checkpoint_controls = gui_cls._set_training_checkpoint_controls

    def _make_count_entry(parent, variable: tk.StringVar):
        return tk.Entry(
            parent,
            textvariable=variable,
            width=6,
            justify="center",
            font=ui.FONT_MONO,
            fg=ui.CLR_TEXT_MAIN,
            bg=ui.CLR_CARD,
            insertbackground=ui.CLR_TEXT_MAIN,
            relief="flat",
            highlightthickness=1,
            highlightbackground=ui.CLR_BORDER,
            highlightcolor=ui.COLOR_BLUE,
            disabledbackground=ui.CLR_CARD,
            disabledforeground=ui.CLR_TEXT_MUTED,
        )

    def _build_test_card(
        self,
        parent,
        *,
        title: str,
        subtitle: str,
        accent: str,
        variable: tk.StringVar,
        button_text: str,
        command,
    ):
        card = tk.Frame(
            parent,
            bg=ui.CLR_PANEL,
            highlightbackground=ui.CLR_BORDER,
            highlightthickness=1,
        )
        tk.Label(
            card,
            text=title,
            font=ui.FONT_BOLD,
            fg=accent,
            bg=ui.CLR_PANEL,
            anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(9, 1))
        tk.Label(
            card,
            text=subtitle,
            font=("Microsoft YaHei UI", 7),
            fg=ui.CLR_TEXT_MUTED,
            bg=ui.CLR_PANEL,
            anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(0, 7))

        controls = tk.Frame(card, bg=ui.CLR_PANEL)
        controls.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Label(
            controls,
            text="Games",
            font=("Microsoft YaHei UI", 8, "bold"),
            fg=ui.CLR_TEXT_MUTED,
            bg=ui.CLR_PANEL,
        ).pack(side=tk.LEFT, padx=(0, 6))
        entry = _make_count_entry(controls, variable)
        entry.pack(side=tk.LEFT, padx=(0, 8), ipady=4)
        button = ui.CyberButton(
            controls,
            button_text,
            command,
            bg_color=accent,
            fg_color="#ffffff",
            height=32,
            padding_x=12,
        )
        button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        return card, entry, button

    def _set_performance_output(self, text: str) -> None:
        pass

    def _build_left_panel(self, parent):
        original_build_left_panel(self, parent)
        old_test_button = getattr(self, "btn_training_test_checkpoint", None)
        old_smoke_button = getattr(self, "btn_training_smoke", None)
        if old_test_button is None or old_smoke_button is None:
            return

        old_test_row = old_test_button.master
        old_smoke_row = old_smoke_button.master
        checkpoint_body = old_test_row.master

        old_test_row.pack_forget()
        old_smoke_row.pack_forget()
        for child in checkpoint_body.winfo_children():
            if child in (old_test_row, old_smoke_row):
                continue
            try:
                text = str(child.cget("text"))
            except Exception:
                continue
            if text.startswith("Load shows the score saved"):
                child.pack_forget()

        header = tk.Frame(checkpoint_body, bg=ui.CLR_CARD)
        header.pack(fill=tk.X, pady=(10, 5))
        tk.Label(
            header,
            text="Validation",
            font=ui.FONT_BOLD,
            fg=ui.COLOR_BLUE,
            bg=ui.CLR_CARD,
        ).pack(side=tk.LEFT)
        tk.Label(
            header,
            text="model evaluation / training-path smoke",
            font=("Microsoft YaHei UI", 7),
            fg=ui.CLR_TEXT_MUTED,
            bg=ui.CLR_CARD,
        ).pack(side=tk.RIGHT)

        grid = tk.Frame(checkpoint_body, bg=ui.CLR_CARD)
        grid.pack(fill=tk.X)
        grid.columnconfigure(0, weight=1, uniform="validation-card")
        grid.columnconfigure(1, weight=1, uniform="validation-card")

        model_card, self.ent_training_eval_episodes, self.btn_training_test_checkpoint = (
            _build_test_card(
                self,
                grid,
                title="Loaded model",
                subtitle="Greedy evaluation · epsilon 0",
                accent=ui.COLOR_GREEN,
                variable=self.var_training_eval_episodes,
                button_text="Test model",
                command=self._test_training_checkpoint,
            )
        )
        model_card.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        smoke_card, self.ent_training_smoke_games, self.btn_training_smoke = _build_test_card(
            self,
            grid,
            title="Smoke path",
            subtitle="No model: training path · loaded model: greedy test",
            accent=ui.COLOR_BLUE,
            variable=self.var_training_smoke_games,
            button_text="Smoke test",
            command=self._run_training_smoke,
        )
        smoke_card.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        try:
            is_dqn = self._selected_training_spec().key == "double_dqn"
        except Exception:
            is_dqn = False
        self._set_training_checkpoint_controls(is_dqn)

    def _set_training_checkpoint_controls(self, dqn: bool):
        original_set_checkpoint_controls(self, dqn)
        smoke_entry = getattr(self, "ent_training_smoke_games", None)
        if smoke_entry is not None:
            smoke_entry.configure(
                state=tk.DISABLED if ((not dqn) or self._training_running) else tk.NORMAL
            )
        eval_entry = getattr(self, "ent_training_eval_episodes", None)
        if eval_entry is not None:
            eval_entry.configure(
                state=(
                    tk.DISABLED
                    if ((not dqn) or self._training_running or not self._training_checkpoint_path)
                    else tk.NORMAL
                )
            )

    gui_cls._set_training_performance_output = _set_performance_output
    gui_cls._build_left_panel = _build_left_panel
    gui_cls._set_training_checkpoint_controls = _set_training_checkpoint_controls
    gui_cls._training_validation_ui_installed = True


__all__ = ["install_training_validation_ui"]
