# -*- coding: utf-8 -*-
"""Keep the right console scrollbar visually identical to the left debug scrollbar."""

from __future__ import annotations

from typing import Any


def install_gui_scrollbar_style_patch(gui_cls: Any, ui: Any) -> None:
    if getattr(gui_cls, "_scrollbar_style_patch_installed", False):
        return

    original_setup_styles = gui_cls._setup_styles

    def _setup_styles_with_matching_scrollbars(self):
        original_setup_styles(self)
        self.style.configure(
            "Console.Vertical.TScrollbar",
            troughcolor=ui.CLR_PANEL,
            background=ui.CLR_BORDER,
            bordercolor=ui.CLR_PANEL,
            darkcolor=ui.CLR_BORDER,
            lightcolor=ui.CLR_BORDER,
            relief="flat",
            borderwidth=0,
            width=10,
        )
        self.style.map(
            "Console.Vertical.TScrollbar",
            background=[("active", ui.COLOR_BLUE), ("pressed", ui.COLOR_BLUE)],
        )

    gui_cls._setup_styles = _setup_styles_with_matching_scrollbars
    gui_cls._scrollbar_style_patch_installed = True
