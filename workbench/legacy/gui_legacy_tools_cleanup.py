# -*- coding: utf-8 -*-
"""Remove obsolete connection/level-jump card from the main GUI.

The daemon already auto-attaches and refreshes runtime state, while LIVE runners
own level switching.  Keep the underlying command methods for compatibility,
but hide the redundant manual card from the workbench.
"""

from __future__ import annotations

import tkinter as tk
from typing import Any


def install_gui_legacy_tools_cleanup(gui_cls: Any) -> None:
    if getattr(gui_cls, "_legacy_tools_cleanup_installed", False):
        return

    original_build_left_panel = gui_cls._build_left_panel

    def _build_left_panel_without_legacy_tools(self, parent):
        original_build_left_panel(self, parent)
        for widget in parent.winfo_children():
            if not isinstance(widget, tk.Canvas):
                continue
            for frame in widget.winfo_children():
                if not isinstance(frame, tk.Frame):
                    continue
                for card in list(frame.winfo_children()):
                    if not isinstance(card, tk.Frame):
                        continue
                    title = next(
                        (
                            child
                            for child in card.winfo_children()
                            if isinstance(child, tk.Label)
                            and child.cget("text") == "连接与关卡工具"
                        ),
                        None,
                    )
                    if title is not None:
                        card.destroy()
                        return

    gui_cls._build_left_panel = _build_left_panel_without_legacy_tools
    gui_cls._legacy_tools_cleanup_installed = True
