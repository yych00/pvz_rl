# -*- coding: utf-8 -*-
"""Show semantic RuntimePhase and raw GameScene in the GUI header."""

from __future__ import annotations

from typing import Any


def install_runtime_phase_gui(gui_cls: Any, ui: Any) -> None:
    if getattr(gui_cls, "_runtime_phase_gui_installed", False):
        return

    original_refresh_status = gui_cls.refresh_status

    def refresh_status_with_runtime_phase(self):
        original_refresh_status(self)
        if not getattr(self.client, "is_loaded", False):
            return
        try:
            on, _ = self.client.is_game_on()
            if not on:
                return
            state = self.client.get_runtime_state()
            if state.get("error"):
                self.badge_ui.config(
                    text="阶段: API 不可用",
                    bg=ui.COLOR_GOLD,
                    fg=ui.CLR_TEXT_DARK,
                )
                return

            phase = str(state.get("phase_name", "UNKNOWN"))
            raw = str(state.get("raw_scene_name", "SCENE_UNKNOWN"))

            if phase in ("BATTLE", "LEVEL_COMPLETE", "AWARD"):
                bg = ui.COLOR_GREEN
                fg = "#ffffff"
            elif phase == "SEED_SELECTION":
                bg = ui.COLOR_BLUE
                fg = ui.CLR_TEXT_DARK
            elif phase in ("DIALOGUE", "LEVEL_INTRO", "LOADING"):
                bg = ui.COLOR_GOLD
                fg = ui.CLR_TEXT_DARK
            elif phase == "LOST":
                bg = ui.COLOR_RED
                fg = "#ffffff"
            else:
                bg = ui.CLR_CARD
                fg = ui.CLR_TEXT_MAIN

            self.badge_ui.config(
                text=f"阶段: {phase} | raw: {raw}",
                bg=bg,
                fg=fg,
            )
        except Exception:
            pass

    gui_cls.refresh_status = refresh_status_with_runtime_phase
    gui_cls._runtime_phase_gui_installed = True
