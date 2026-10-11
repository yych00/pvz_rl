# -*- coding: utf-8 -*-
"""Tabbed left workspace for Adventure LIVE, Gym Training and Native Debug.

The left side uses a fixed tab header plus one dedicated viewport. Existing LIVE,
training and debug widgets are still built exactly once inside that viewport, so
status, scroll position and test state survive tab switches. Mouse-wheel input is
routed to the currently visible tab instead of letting multiple Canvas widgets
install competing global wheel handlers.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Optional


def install_gui_left_tabs(gui_cls: Any, ui: Any) -> None:
    if getattr(gui_cls, "_left_tabs_installed", False):
        return

    original_build_left_panel = gui_cls._build_left_panel

    def _label_with_text(widget: tk.Misc, text: str) -> Optional[tk.Label]:
        for child in widget.winfo_children():
            if isinstance(child, tk.Label) and str(child.cget("text")) == text:
                return child
            found = _label_with_text(child, text)
            if found is not None:
                return found
        return None

    def _is_descendant(widget: tk.Misc | None, ancestor: tk.Misc | None) -> bool:
        node = widget
        while node is not None:
            if node is ancestor:
                return True
            node = getattr(node, "master", None)
        return False

    def _set_tab_visual(self, active: str) -> None:
        self._left_tab_active = active
        for name, parts in self._left_tab_widgets.items():
            selected = name == active
            parts["label"].config(
                fg=ui.COLOR_BLUE if selected else ui.CLR_TEXT_MUTED,
                bg=ui.CLR_BG,
            )
            parts["underline"].config(
                bg=ui.COLOR_BLUE if selected else ui.CLR_BG,
            )

    def _active_left_canvas(self):
        return getattr(self, "_left_scroll_canvases", {}).get(
            getattr(self, "_left_tab_active", "live")
        )

    def _route_left_mousewheel(self, event):
        viewport = getattr(self, "_left_viewport", None)
        if viewport is None or not _is_descendant(getattr(event, "widget", None), viewport):
            return None

        canvas = self._active_left_canvas()
        if canvas is None or not canvas.winfo_exists():
            return None

        bbox = canvas.bbox("all")
        canvas_h = canvas.winfo_height()
        content_h = (bbox[3] - bbox[1]) if bbox else 0

        # 内容未超出视口（界面放得下），彻底禁止滚动并牢牢对齐顶部 (0, 0)
        if content_h <= canvas_h or canvas_h <= 1:
            canvas.yview_moveto(0.0)
            return "break"

        delta = int(getattr(event, "delta", 0) or 0)
        if delta == 0:
            return None
        units = int(-delta / 120)
        if units == 0:
            units = -1 if delta > 0 else 1

        y_first, y_last = canvas.yview()

        # 向上滚轮（试图向下平移内容）：若已在顶端，坚决阻止负向位移拉出上方空白
        if units < 0:
            if y_first <= 0.001:
                canvas.yview_moveto(0.0)
                return "break"
        # 向下滚轮（试图向上平移内容）：若已在底端，阻止继续向下扯出多余黑底
        elif units > 0:
            if y_last >= 0.999:
                return "break"

        canvas.yview_scroll(units, "units")

        # 滚动后二次防御：若发生负向越界立即归零锁定在顶部
        post_first, _ = canvas.yview()
        if post_first < 0.0:
            canvas.yview_moveto(0.0)
        return "break"

    def _switch_left_tab(self, name: str) -> None:
        if name not in ("live", "training", "debug"):
            return

        live_panel = getattr(self, "_left_live_panel", None)
        debug_canvas = getattr(self, "_left_debug_canvas", None)
        debug_scrollbar = getattr(self, "_left_debug_scrollbar", None)
        training_panel = getattr(self, "_left_training_panel", None)
        if live_panel is None or training_panel is None or debug_canvas is None or debug_scrollbar is None:
            return

        live_panel.pack_forget()
        training_panel.pack_forget()
        debug_canvas.pack_forget()
        debug_scrollbar.pack_forget()

        if name == "live":
            live_panel.pack(fill=tk.BOTH, expand=True)
        elif name == "training":
            training_panel.pack(fill=tk.BOTH, expand=True)
        else:
            debug_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            debug_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._set_left_tab_visual(name)
        active_canvas = self._active_left_canvas()
        if active_canvas and active_canvas.winfo_exists():
            active_canvas.yview_moveto(0.0)

    def _apply_taller_workbench_geometry(self) -> None:
        """Give the workbench more vertical room without exceeding the desktop."""
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w = min(1680, max(1260, int(screen_w * 0.90)))
        usable_h = max(760, screen_h - 72)
        win_h = min(1040, max(820, int(screen_h * 0.94)), usable_h)
        pos_x = max(0, (screen_w - win_w) // 2)
        pos_y = max(0, (screen_h - win_h) // 2)
        self.root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        self.root.minsize(1180, 760)

    def _build_left_panel_with_tabs(self, parent):
        # Keep tab controls outside the scrolling area. Everything that used to
        # be built directly under the left pane now lives in one viewport below
        # the tabs, matching the right side's fixed chrome + scrolling content
        # structure and preventing pack geometry from competing across tabs.
        tabs = tk.Frame(parent, bg=ui.CLR_BG, height=42)
        # Keep every workspace attached to the common navigation strip.
        tabs.pack(side=tk.TOP, fill=tk.X)
        tabs.pack_propagate(False)

        viewport = tk.Frame(parent, bg=ui.CLR_BG)
        viewport.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._left_viewport = viewport

        original_build_left_panel(self, viewport)

        children = list(viewport.winfo_children())
        debug_canvas = next(
            (child for child in children if isinstance(child, tk.Canvas)),
            None,
        )
        debug_scrollbar = next(
            (child for child in children if isinstance(child, ttk.Scrollbar)),
            None,
        )
        debug_header = next(
            (
                child
                for child in children
                if isinstance(child, tk.Frame)
                and _label_with_text(child, "Native Debug") is not None
            ),
            None,
        )
        live_title = _label_with_text(viewport, "Adventure LIVE")
        live_panel = None
        if live_title is not None:
            node = live_title
            while node.master is not None and node.master is not viewport:
                node = node.master
            if node.master is viewport:
                live_panel = node

        if live_panel is None or debug_canvas is None or debug_scrollbar is None:
            return
        training_panel = getattr(self, "_training_panel", None)
        if training_panel is None:
            return

        # The tab header replaces both old section headings.
        if live_title is not None:
            live_title.pack_forget()
        if debug_header is not None:
            # The tab strip now owns this label. Removing the legacy header
            # prevents it from reserving space in the shared viewport.
            debug_header.destroy()

        live_panel.pack_forget()
        training_panel.pack_forget()
        debug_canvas.pack_forget()
        debug_scrollbar.pack_forget()

        self._left_tab_widgets = {}
        for name, text in (
            ("live", "Adventure LIVE"),
            ("training", "Gym Training"),
            ("debug", "Native Debug"),
        ):
            item = tk.Frame(tabs, bg=ui.CLR_BG)
            item.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))

            label = tk.Label(
                item,
                text=text,
                font=("Microsoft YaHei UI", 10, "bold"),
                fg=ui.CLR_TEXT_MUTED,
                bg=ui.CLR_BG,
                padx=14,
                pady=7,
                cursor="hand2",
            )
            label.pack(side=tk.TOP, fill=tk.X)
            underline = tk.Frame(item, bg=ui.CLR_BG, height=2)
            underline.pack(side=tk.BOTTOM, fill=tk.X)

            label.bind("<Button-1>", lambda _e, tab=name: self._switch_left_tab(tab))
            item.bind("<Button-1>", lambda _e, tab=name: self._switch_left_tab(tab))
            underline.bind("<Button-1>", lambda _e, tab=name: self._switch_left_tab(tab))
            self._left_tab_widgets[name] = {
                "item": item,
                "label": label,
                "underline": underline,
            }

        training_canvas = next(
            (
                child
                for child in training_panel.winfo_children()
                if isinstance(child, tk.Canvas)
            ),
            None,
        )
        live_canvas = getattr(self, "_live_canvas", None)

        # The old LIVE/Debug builders installed bind_all handlers on Canvas
        # enter/leave. Remove those hooks and route wheel input once at the
        # toplevel so Training gets identical behavior and the right console is
        # unaffected whenever the pointer is outside the left viewport.
        for canvas in (live_canvas, debug_canvas):
            if canvas is not None:
                canvas.unbind("<Enter>")
                canvas.unbind("<Leave>")

        self._left_live_panel = live_panel
        self._left_training_panel = training_panel
        self._left_debug_canvas = debug_canvas
        self._left_debug_scrollbar = debug_scrollbar
        self._left_scroll_canvases = {
            "live": live_canvas,
            "training": training_canvas,
            "debug": debug_canvas,
        }
        self._left_tab_active = "live"
        self.root.bind("<MouseWheel>", self._route_left_mousewheel, add="+")
        self._switch_left_tab("live")
        self._apply_taller_workbench_geometry()

    gui_cls._set_left_tab_visual = _set_tab_visual
    gui_cls._active_left_canvas = _active_left_canvas
    gui_cls._route_left_mousewheel = _route_left_mousewheel
    gui_cls._switch_left_tab = _switch_left_tab
    gui_cls._apply_taller_workbench_geometry = _apply_taller_workbench_geometry
    gui_cls._build_left_panel = _build_left_panel_with_tabs
    gui_cls._left_tabs_installed = True
