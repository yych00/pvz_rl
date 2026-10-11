# -*- coding: utf-8 -*-
"""PvZ Gym native test workbench.

The primary GUI surface is a 5-scene / 50-level LIVE matrix.  A user selects a
single Adventure level and runs E2E or E2E x5, or runs a one-pass test over all
50 levels.  Playthrough mode is inferred from game memory when a run starts and
then held constant for that run.  The bottom control bar owns pause / continue /
stop for every LIVE workflow.

Legacy BattleAction/legality GUI extensions still mount into the debug scroll
area through ``_build_left_panel`` and ``_create_card``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional

try:
    import ctypes
    ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
except Exception:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass

from .c_interface import PvzGymNativeDll
from .early_levels_live_tests import (
    PLAYTHROUGH_1,
    PLAYTHROUGH_2,
    SIMULATION_ANIMATED,
    SIMULATION_CONTINUOUS,
    SIMULATION_HEADLESS,
    SIMULATION_STEPPED,
    run_early_level_e2e,
    run_early_level_stress,
)
from .level_test_registry import SCENES, all_level_specs, get_level_spec, get_scene_specs
from .test_cases import NativeUnitTestSuite, TestCaseResult
from .deck_panel import DeckPanel, start_deck_test

logger = logging.getLogger("pvz_gym_unit_test.gui")

# -----------------------------------------------------------------------------
# Theme
# -----------------------------------------------------------------------------
CLR_BG = "#0d1117"
CLR_PANEL = "#161b22"
CLR_CARD = "#21262d"
CLR_BORDER = "#30363d"
CLR_BORDER_ACTIVE = "#58a6ff"
CLR_TEXT_MAIN = "#f0f6fc"
CLR_TEXT_MUTED = "#8b949e"
CLR_TEXT_DARK = "#0d1117"

COLOR_BLUE = "#58a6ff"
COLOR_GREEN = "#2ea043"
# Kept for compatibility with extension modules, but intentionally neutralized.
COLOR_GOLD = "#64748b"
COLOR_RED = "#f85149"
COLOR_BTN_HOVER = "#ffffff"

FONT_MAIN = ("Microsoft YaHei UI", 9)
FONT_BOLD = ("Microsoft YaHei UI", 9, "bold")
FONT_TITLE = ("Microsoft YaHei UI", 10, "bold")
FONT_MONO = ("Consolas", 9)

LIVE_STATUS_COLORS = {
    "READY": CLR_TEXT_MUTED,
    "RUNNING": COLOR_BLUE,
    "PAUSED": "#94a3b8",
    "PASS": COLOR_GREEN,
    "FAIL": COLOR_RED,
    "STOPPED": CLR_TEXT_MUTED,
}

LIVE_SPEED_CONFIGS = {
    "连续运行 · 不暂停（原速）": {
        "simulation_mode": SIMULATION_CONTINUOUS,
        "animated_frame_ms": 10,
    },
    "同步步进 · 约10x": {
        "simulation_mode": SIMULATION_STEPPED,
        "animated_frame_ms": 10,
    },
    "动画原生 · 1x": {
        "simulation_mode": SIMULATION_ANIMATED,
        "animated_frame_ms": 10,
    },
    "动画原生 · 2x": {
        "simulation_mode": SIMULATION_ANIMATED,
        "animated_frame_ms": 5,
    },
    "动画原生 · 5x": {
        "simulation_mode": SIMULATION_ANIMATED,
        "animated_frame_ms": 2,
    },
    "动画原生 · 10x": {
        "simulation_mode": SIMULATION_ANIMATED,
        "animated_frame_ms": 1,
    },
    "无头持久 · 最快": {
        "simulation_mode": SIMULATION_HEADLESS,
        "animated_frame_ms": 1,
    },
}


class CyberButton(tk.Canvas):
    """Small dark-theme adaptive button used by the native workbench."""

    def __init__(
        self,
        parent,
        text: str,
        command,
        bg_color: str = COLOR_BLUE,
        fg_color: str = CLR_TEXT_DARK,
        width: Optional[int] = None,
        height: int = 32,
        font=FONT_BOLD,
        cursor: str = "hand2",
        padding_x: int = 16,
    ):
        try:
            measure_font = tkfont.Font(font=font)
            text_pixel_width = measure_font.measure(text)
        except Exception:
            text_pixel_width = len(text) * 14
        auto_width = text_pixel_width + padding_x * 2
        initial_width = max(width or 0, auto_width)
        super().__init__(
            parent,
            width=initial_width,
            height=height,
            bg=parent["bg"],
            highlightthickness=0,
            cursor=cursor,
        )
        self.text = text
        self.command = command
        self.base_color = bg_color
        self.fg_color = fg_color
        self.w = initial_width
        self.h = height
        self.font = font
        self.is_hover = False
        self.is_disabled = False
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self.bind("<Configure>", self._on_configure)
        self._render()

    def _on_configure(self, event):
        if event.width > 1 and event.height > 1:
            if self.w != event.width or self.h != event.height:
                self.w = event.width
                self.h = event.height
                self._render()

    def _render(self):
        self.delete("all")
        if self.is_disabled:
            fill_clr = CLR_CARD
            txt_clr = "#6e7681"
        elif self.is_hover:
            fill_clr = COLOR_BTN_HOVER
            txt_clr = CLR_TEXT_DARK
        else:
            fill_clr = self.base_color
            txt_clr = self.fg_color
        r = 4
        self.create_polygon(
            r, 0, self.w - r, 0, self.w, r, self.w, self.h - r,
            self.w - r, self.h, r, self.h, 0, self.h - r, 0, r,
            fill=fill_clr, outline="", smooth=True,
        )
        self.create_text(
            self.w // 2,
            self.h // 2,
            text=self.text,
            fill=txt_clr,
            font=self.font,
        )

    def _on_enter(self, _event):
        if not self.is_disabled:
            self.is_hover = True
            self._render()

    def _on_leave(self, _event):
        if not self.is_disabled:
            self.is_hover = False
            self._render()

    def _on_click(self, _event):
        if not self.is_disabled and self.command:
            self.command()

    def set_disabled(self, disabled: bool):
        self.is_disabled = bool(disabled)
        self._render()

    def set_text(self, new_text: str):
        self.text = str(new_text)
        self._render()

    def set_colors(self, bg_color: str, fg_color: str):
        self.base_color = bg_color
        self.fg_color = fg_color
        self._render()


class PvzGymUnitTestGUI:
    """PvZ Gym C++ native workbench + 50-level LIVE test controller."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PvZ Gym - LIVE & Training Workbench")
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w = min(1680, max(1260, int(screen_w * 0.90)))
        win_h = min(980, max(760, int(screen_h * 0.88)))
        pos_x = max(0, (screen_w - win_w) // 2)
        pos_y = max(0, (screen_h - win_h) // 2)
        self.root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        self.root.minsize(1180, 700)
        self.root.configure(bg=CLR_BG)

        self.client = PvzGymNativeDll()
        if self.client.is_loaded:
            self.client.set_auto_dialogue(True)
        self.suite = NativeUnitTestSuite(self.client)

        self._live_scene = 1
        self._live_selected_level = 1
        self._live_status: Dict[int, Dict[str, str]] = {
            i: {"status": "READY", "detail": ""} for i in range(1, 51)
        }
        self._live_running = False
        self._live_current_level: Optional[int] = None
        self._live_pause_event: Optional[threading.Event] = None
        self._live_stop_event: Optional[threading.Event] = None
        self._live_row_widgets: Dict[int, Dict[str, Any]] = {}
        self._scene_buttons: Dict[int, CyberButton] = {}
        self._static_test_running = False
        self._is_topmost = False
        self.btn_topmost: Optional[CyberButton] = None
        self._auto_dialogue_enabled = (
            self.client.get_auto_dialogue() if self.client.is_loaded else False
        )
        self._auto_tutorial_enabled = True
        self._no_dave_enabled = True
        self._auto_cancel_intro_enabled = True
        self._auto_whack_enabled = True
        self.btn_toggle_auto_dialogue: Optional[CyberButton] = None
        self.btn_toggle_auto_tutorial: Optional[CyberButton] = None
        self.btn_toggle_no_dave: Optional[CyberButton] = None
        self.btn_toggle_auto_intro: Optional[CyberButton] = None
        self._intro_stage1_triggered = False
        self._had_seed_selection = False
        self._intro_stage2_triggered = False

        self._setup_styles()
        self._build_ui()
        self.refresh_status()
        self._start_daemon_loop()

    # ------------------------------------------------------------------ styles
    def _setup_styles(self):
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception as exc:
            logger.debug("theme_use(clam): %s", exc)

        self.style.configure(
            "Dark.Vertical.TScrollbar",
            troughcolor=CLR_PANEL,
            background="#484f58",
            bordercolor=CLR_PANEL,
            darkcolor="#484f58",
            lightcolor="#484f58",
            relief="flat",
            borderwidth=0,
            width=12,
        )
        self.style.map(
            "Dark.Vertical.TScrollbar",
            background=[("active", COLOR_BLUE), ("pressed", COLOR_BLUE)],
        )
        self.style.configure(
            "Console.Vertical.TScrollbar",
            troughcolor="#050811",
            background="#3b4252",
            bordercolor="#050811",
            darkcolor="#3b4252",
            lightcolor="#3b4252",
            relief="flat",
            borderwidth=0,
            width=10,
        )
        self.style.map(
            "Console.Vertical.TScrollbar",
            background=[("active", COLOR_BLUE), ("pressed", COLOR_BLUE)],
        )
        self.style.configure(
            "Horizontal.TProgressbar",
            troughcolor=CLR_PANEL,
            background=COLOR_BLUE,
            bordercolor=CLR_BORDER,
            lightcolor=COLOR_BLUE,
            darkcolor=COLOR_BLUE,
            thickness=4,
        )
        self.style.configure(
            "TCombobox",
            fieldbackground=CLR_PANEL,
            background=CLR_CARD,
            foreground=CLR_TEXT_MAIN,
            bordercolor=CLR_BORDER,
            arrowcolor=COLOR_BLUE,
            padding=3,
        )
        self.style.map(
            "TCombobox",
            fieldbackground=[("readonly", CLR_PANEL)],
            selectbackground=[("readonly", "#1f6feb")],
            selectforeground=[("readonly", "#ffffff")],
        )
        self.root.option_add("*TCombobox*Listbox.background", CLR_PANEL)
        self.root.option_add("*TCombobox*Listbox.foreground", CLR_TEXT_MAIN)
        self.root.option_add("*TCombobox*Listbox.selectBackground", "#1f6feb")
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")

    # --------------------------------------------------------------------- UI
    def _build_ui(self):
        self._build_header()

        self.main_paned = tk.PanedWindow(
            self.root,
            orient=tk.HORIZONTAL,
            bg=CLR_BG,
            bd=0,
            sashwidth=4,
            sashrelief="flat",
            showhandle=False,
        )
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=12, pady=(6, 10))

        left_container = tk.Frame(self.main_paned, bg=CLR_BG, width=780)
        right_container = tk.Frame(self.main_paned, bg=CLR_BG)
        self.main_paned.add(left_container, minsize=620)
        self.main_paned.add(right_container, minsize=480)

        self._build_left_panel(left_container)
        self._build_right_console(right_container)

        self.root.update_idletasks()
        try:
            self.main_paned.sash_place(0, 780, 0)
        except Exception:
            pass

    def _build_header(self):
        frame = tk.Frame(self.root, bg=CLR_PANEL, height=86)
        frame.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(10, 0))
        frame.pack_propagate(False)

        title_row = tk.Frame(frame, bg=CLR_PANEL, height=48)
        title_row.pack(side=tk.TOP, fill=tk.X)
        title_row.pack_propagate(False)

        title_box = tk.Frame(title_row, bg=CLR_PANEL)
        title_box.pack(side=tk.LEFT, padx=14, pady=6)
        tk.Label(
            title_box,
            text="[PvZ Gym] LIVE & Training Workbench",
            font=("Microsoft YaHei UI", 12, "bold"),
            fg=COLOR_BLUE,
            bg=CLR_PANEL,
        ).pack(anchor="w")
        tk.Label(
            title_box,
            text="Adventure LIVE · Gym Training · 权威 C++ 状态与动作掩码",
            font=("Microsoft YaHei UI", 8),
            fg=CLR_TEXT_MUTED,
            bg=CLR_PANEL,
        ).pack(anchor="w")

        CyberButton(
            title_row,
            "刷新运行状态",
            self.refresh_status,
            bg_color=CLR_CARD,
            fg_color=CLR_TEXT_MAIN,
            height=28,
            padding_x=12,
        ).pack(side=tk.RIGHT, padx=(6, 14), pady=10)

        self.btn_topmost = CyberButton(
            title_row,
            "游戏置顶: 关",
            self.toggle_game_topmost,
            bg_color=CLR_CARD,
            fg_color=CLR_TEXT_MUTED,
            height=28,
            padding_x=12,
        )
        self.btn_topmost.pack(side=tk.RIGHT, padx=6, pady=10)

        # Runtime facts have a dedicated row so no badge needs truncation.
        badges = tk.Frame(frame, bg=CLR_BG, height=30)
        badges.pack(side=tk.TOP, fill=tk.X)
        badges.pack_propagate(False)
        status_items = tk.Frame(badges, bg=CLR_BG)
        status_items.pack(side=tk.RIGHT, padx=11)
        self.badge_dll = self._badge(status_items, "DLL: 检测中")
        self.badge_game = self._badge(status_items, "进程: 探测中")
        self.badge_ui = self._badge(status_items, "阶段: --")
        self.badge_level = self._badge(status_items, "关卡: --")
        self.badge_clock = self._badge(status_items, "时钟: 0")
        self.badge_paused = self._badge(status_items, "暂停: --")

    def _badge(self, parent, text):
        label = tk.Label(
            parent,
            text=text,
            font=FONT_BOLD,
            fg=CLR_TEXT_MAIN,
            bg=CLR_CARD,
            padx=8,
            pady=3,
        )
        label.pack(side=tk.LEFT, padx=3)
        return label

    def _build_left_panel(self, parent):
        # LIVE panel is fixed; old/manual debug cards stay in a scrollable area
        # below it. Extension modules still discover this direct child Canvas.
        self._build_live_panel(parent)

        debug_header = tk.Frame(parent, bg=CLR_BG)
        debug_header.pack(fill=tk.X, pady=(8, 4))
        tk.Label(
            debug_header,
            text="Native Debug",
            font=FONT_BOLD,
            fg=CLR_TEXT_MUTED,
            bg=CLR_BG,
        ).pack(side=tk.LEFT)

        canvas = tk.Canvas(parent, bg=CLR_BG, bd=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            parent,
            orient="vertical",
            style="Dark.Vertical.TScrollbar",
            command=canvas.yview,
        )
        scrollable_frame = tk.Frame(canvas, bg=CLR_BG)

        # Keep Native Debug's viewport behavior identical to Adventure LIVE.
        # A plain bbox-based scrollregion lets a short debug panel move away
        # from the top of the Canvas and exposes a blank band above it.
        last_bbox = [None]

        def _sync_debug_scrollregion(canvas_h=None):
            bbox = canvas.bbox("all")
            if not bbox:
                return
            if bbox == last_bbox[0] and canvas_h is None:
                return
            last_bbox[0] = bbox
            viewport_h = canvas.winfo_height() if canvas_h is None else canvas_h
            content_h = bbox[3] - bbox[1]
            if content_h <= viewport_h:
                canvas.configure(scrollregion=(0, 0, bbox[2], viewport_h))
                canvas.yview_moveto(0.0)
            else:
                canvas.configure(scrollregion=(0, 0, bbox[2], bbox[3]))

        scrollable_frame.bind("<Configure>", lambda _e: _sync_debug_scrollregion())
        window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        last_width = [0]

        def _on_debug_canvas_configure(event):
            if event.width > 1 and event.width != last_width[0]:
                last_width[0] = event.width
                canvas.itemconfig(window, width=event.width)
            _sync_debug_scrollregion(event.height)

        canvas.bind("<Configure>", _on_debug_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def _wheel(event):
            bbox = canvas.bbox("all")
            content_h = (bbox[3] - bbox[1]) if bbox else 0
            if content_h <= canvas.winfo_height():
                canvas.yview_moveto(0.0)
                return "break"
            units = int(-1 * (event.delta / 120))
            if units:
                canvas.yview_scroll(units, "units")
            return "break"

        canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", _wheel))
        canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))

        self._build_native_debug_cards(scrollable_frame)

    def _build_live_panel(self, parent):
        container = tk.Frame(parent, bg=CLR_BG)
        container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(container, bg=CLR_BG, bd=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            container,
            orient="vertical",
            style="Dark.Vertical.TScrollbar",
            command=canvas.yview,
        )
        scrollable_frame = tk.Frame(canvas, bg=CLR_BG)

        # 尺寸防抖：仅当内部组件总范围真正改变时才更新滚动区域
        last_bbox = [None]

        def _on_frame_configure(_event):
            bbox = canvas.bbox("all")
            if bbox and bbox != last_bbox[0]:
                last_bbox[0] = bbox
                content_h = bbox[3] - bbox[1]
                canvas_h = canvas.winfo_height()
                # 严格按照内容真实范围设置 scrollregion，杜绝虚增高度产生的底部空当
                if content_h <= canvas_h:
                    canvas.configure(scrollregion=(0, 0, bbox[2], canvas_h))
                    canvas.yview_moveto(0.0)
                else:
                    canvas.configure(scrollregion=(0, 0, bbox[2], bbox[3]))

        scrollable_frame.bind("<Configure>", _on_frame_configure)
        window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        # 宽度防抖：仅当 Canvas 宽度真正改变时更新 window 宽度，避免微小重绘陷入循环震荡
        last_width = [0]

        def _on_canvas_configure(event):
            if event.width > 1 and event.width != last_width[0]:
                last_width[0] = event.width
                canvas.itemconfig(window, width=event.width)
            bbox = canvas.bbox("all")
            if bbox:
                content_h = bbox[3] - bbox[1]
                if content_h <= event.height:
                    canvas.configure(scrollregion=(0, 0, bbox[2], event.height))
                    canvas.yview_moveto(0.0)
                else:
                    canvas.configure(scrollregion=(0, 0, bbox[2], bbox[3]))

        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def _wheel(event):
            bbox = canvas.bbox("all")
            canvas_h = canvas.winfo_height()
            content_h = (bbox[3] - bbox[1]) if bbox else 0
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
            if units < 0 and y_first <= 0.001:
                canvas.yview_moveto(0.0)
                return "break"
            if units > 0 and y_last >= 0.999:
                return "break"
            canvas.yview_scroll(units, "units")
            if canvas.yview()[0] < 0.0:
                canvas.yview_moveto(0.0)
            return "break"

        canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", _wheel))
        canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))

        self._live_canvas = canvas
        self._live_scrollable_frame = scrollable_frame

        outer = tk.Frame(
            scrollable_frame,
            bg=CLR_PANEL,
            highlightbackground=CLR_BORDER,
            highlightthickness=1,
        )
        outer.pack(fill=tk.X)

        top = tk.Frame(outer, bg=CLR_PANEL)
        top.pack(fill=tk.X, padx=12, pady=(10, 8))
        tk.Label(
            top,
            text="Adventure LIVE",
            font=FONT_TITLE,
            fg=CLR_TEXT_MAIN,
            bg=CLR_PANEL,
        ).pack(side=tk.LEFT)
        self.lbl_live_summary = tk.Label(
            top,
            text="50 关就绪",
            font=FONT_BOLD,
            fg=CLR_TEXT_MUTED,
            bg=CLR_PANEL,
        )
        self.lbl_live_summary.pack(side=tk.RIGHT)

        body = tk.Frame(outer, bg=CLR_PANEL)
        body.pack(fill=tk.X, padx=12)

        scene_box = tk.Frame(body, bg=CLR_PANEL, width=145)
        scene_box.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        scene_box.pack_propagate(False)
        tk.Label(
            scene_box,
            text="场景",
            font=FONT_BOLD,
            fg=CLR_TEXT_MUTED,
            bg=CLR_PANEL,
        ).pack(anchor="w", pady=(0, 6))
        for area, name, _scene in SCENES:
            btn = CyberButton(
                scene_box,
                name,
                lambda a=area: self._select_live_scene(a),
                bg_color=COLOR_BLUE if area == self._live_scene else CLR_CARD,
                fg_color="#ffffff" if area == self._live_scene else CLR_TEXT_MAIN,
                height=32,
                padding_x=10,
            )
            btn.pack(fill=tk.X, pady=2)
            self._scene_buttons[area] = btn

        list_box = tk.Frame(body, bg=CLR_PANEL)
        list_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        header = tk.Frame(list_box, bg=CLR_CARD)
        header.pack(fill=tk.X, pady=(0, 3))
        tk.Label(header, text="", width=3, bg=CLR_CARD).pack(side=tk.LEFT)
        tk.Label(header, text="关卡", width=8, anchor="w", font=FONT_BOLD, fg=CLR_TEXT_MUTED, bg=CLR_CARD).pack(side=tk.LEFT)
        tk.Label(header, text="测试状态", width=14, anchor="w", font=FONT_BOLD, fg=CLR_TEXT_MUTED, bg=CLR_CARD).pack(side=tk.LEFT)
        tk.Label(header, text="结果", anchor="w", font=FONT_BOLD, fg=CLR_TEXT_MUTED, bg=CLR_CARD).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.live_rows_frame = tk.Frame(list_box, bg=CLR_PANEL)
        self.live_rows_frame.pack(fill=tk.X)
        self._render_live_rows()

        selected_bar = tk.Frame(outer, bg=CLR_CARD)
        selected_bar.pack(fill=tk.X, padx=12, pady=(10, 6))
        selected_bar.columnconfigure(0, weight=1)
        selected_bar.columnconfigure(1, weight=0)
        self.lbl_live_selected = tk.Label(
            selected_bar,
            text="已选择 1-1",
            font=FONT_BOLD,
            fg=COLOR_BLUE,
            bg=CLR_CARD,
        )
        self.lbl_live_selected.grid(row=0, column=0, sticky="w", padx=10, pady=8)
        self.lbl_live_current = tk.Label(
            selected_bar,
            text="READY",
            font=FONT_BOLD,
            fg=CLR_TEXT_MUTED,
            bg=CLR_CARD,
        )
        self.lbl_live_current.grid(row=0, column=1, sticky="e", padx=10, pady=8)

        speed_bar = tk.Frame(outer, bg=CLR_PANEL)
        speed_bar.pack(fill=tk.X, padx=12, pady=(0, 8))
        speed_bar.columnconfigure(1, weight=1)
        tk.Label(
            speed_bar,
            text="推进速率",
            font=FONT_BOLD,
            fg=CLR_TEXT_MUTED,
            bg=CLR_PANEL,
        ).grid(row=0, column=0, sticky="w", padx=(0, 10), pady=2)
        self.var_live_speed = tk.StringVar(value=next(iter(LIVE_SPEED_CONFIGS)))
        self.cmb_live_speed = ttk.Combobox(
            speed_bar,
            textvariable=self.var_live_speed,
            values=tuple(LIVE_SPEED_CONFIGS),
            state="readonly",
            width=24,
        )
        self.cmb_live_speed.grid(row=0, column=1, sticky="ew", pady=2)

        tk.Label(speed_bar, text="测试周目", font=FONT_BOLD, fg=CLR_TEXT_MUTED,
                 bg=CLR_PANEL).grid(row=1, column=0, sticky="w", pady=(6, 2))
        self.var_live_deck_playthrough = tk.StringVar(value="1")
        self.cmb_live_playthrough = ttk.Combobox(speed_bar, textvariable=self.var_live_deck_playthrough,
                                                values=("1", "2"), state="readonly", width=24)
        self.cmb_live_playthrough.grid(row=1, column=1, sticky="ew", pady=(6, 2))
        self.live_deck_panel = DeckPanel(
            outer, sys.modules[__name__],
            lambda: (self._live_selected_level, int(self.var_live_deck_playthrough.get())),
            lambda panel: start_deck_test(self, panel))
        self.live_deck_panel.pack(fill=tk.X, padx=12, pady=(0, 10))
        self.var_live_deck_playthrough.trace_add("write", lambda *_: self.live_deck_panel.refresh())

        actions = tk.Frame(outer, bg=CLR_PANEL)
        actions.pack(fill=tk.X, padx=12, pady=(0, 6))
        for col in range(3):
            actions.columnconfigure(col, weight=1)
        self.btn_live_e2e = CyberButton(
            actions,
            "测试所选 E2E",
            lambda: self._start_selected_live(1),
            bg_color=COLOR_BLUE,
            fg_color="#ffffff",
            height=36,
        )
        self.btn_live_e2e.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        self.btn_live_x5 = CyberButton(
            actions,
            "测试所选 x5",
            lambda: self._start_selected_live(5),
            bg_color=CLR_CARD,
            fg_color=COLOR_BLUE,
            height=36,
        )
        self.btn_live_x5.grid(row=0, column=1, padx=4, sticky="ew")
        self.btn_live_all = CyberButton(
            actions,
            "一键测试全部 50 关",
            self._start_all_live,
            bg_color=COLOR_GREEN,
            fg_color="#ffffff",
            height=36,
        )
        self.btn_live_all.grid(row=0, column=2, padx=(4, 0), sticky="ew")

        # Fixed bottom controls requested for every LIVE workflow.
        controls = tk.Frame(outer, bg=CLR_PANEL)
        controls.pack(fill=tk.X, padx=12, pady=(0, 12))
        for col in range(3):
            controls.columnconfigure(col, weight=1)
        self.btn_live_pause = CyberButton(
            controls,
            "暂停",
            self._pause_live,
            bg_color=CLR_CARD,
            fg_color=CLR_TEXT_MAIN,
            height=32,
        )
        self.btn_live_pause.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        self.btn_live_continue = CyberButton(
            controls,
            "继续",
            self._continue_live,
            bg_color=CLR_CARD,
            fg_color=COLOR_BLUE,
            height=32,
        )
        self.btn_live_continue.grid(row=0, column=1, padx=4, sticky="ew")
        self.btn_live_stop = CyberButton(
            controls,
            "终止",
            self._stop_live,
            bg_color=CLR_CARD,
            fg_color=COLOR_RED,
            height=32,
        )
        self.btn_live_stop.grid(row=0, column=2, padx=(4, 0), sticky="ew")
        self._set_live_controls_running(False)

    def _build_native_debug_cards(self, frame):
        # Use the same page-level hierarchy as Adventure LIVE and Gym Training:
        # identity first, then a concise description, then the actionable tools.
        overview = tk.Frame(
            frame,
            bg=CLR_PANEL,
            highlightbackground=CLR_BORDER,
            highlightthickness=1,
        )
        overview.pack(fill=tk.X, pady=(0, 10))
        overview_top = tk.Frame(overview, bg=CLR_PANEL)
        overview_top.pack(fill=tk.X, padx=12, pady=(10, 4))
        tk.Label(
            overview_top,
            text="Native Debug",
            font=FONT_TITLE,
            fg=CLR_TEXT_MAIN,
            bg=CLR_PANEL,
        ).pack(side=tk.LEFT)
        tk.Label(
            overview_top,
            text="本地 C++ 工具",
            font=FONT_BOLD,
            fg=CLR_TEXT_MUTED,
            bg=CLR_PANEL,
        ).pack(side=tk.RIGHT)
        tk.Label(
            overview,
            text="运行原生单元测试、连接游戏进程，或按指定场景与关卡跳转。所有操作和 Adventure LIVE、Gym Training 相互独立。",
            font=FONT_MAIN,
            fg=CLR_TEXT_MUTED,
            bg=CLR_PANEL,
            anchor="w",
            justify=tk.LEFT,
            wraplength=740,
        ).pack(fill=tk.X, padx=12, pady=(0, 10))

        card = self._create_card(frame, "C++ 原生单元测试")
        tk.Label(
            card,
            text="保留 ABI / 边界 / 原生事实源测试；LIVE 50 关测试与这里相互独立。",
            font=("Microsoft YaHei UI", 8),
            fg=CLR_TEXT_MUTED,
            bg=CLR_PANEL,
        ).pack(anchor="w", padx=12, pady=(0, 8))
        row = tk.Frame(card, bg=CLR_PANEL)
        row.pack(fill=tk.X, padx=12, pady=(0, 12))
        row.columnconfigure(0, weight=2)
        row.columnconfigure(1, weight=1)
        self.btn_run_all = CyberButton(
            row,
            "运行全部 C++ 单元测试",
            self.run_all_tests_async,
            bg_color=COLOR_GREEN,
            fg_color="#ffffff",
            height=34,
        )
        self.btn_run_all.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        CyberButton(
            row,
            "清空日志",
            self.clear_console,
            bg_color=CLR_CARD,
            fg_color=CLR_TEXT_MAIN,
            height=34,
        ).grid(row=0, column=1, padx=(4, 0), sticky="ew")

        card2 = self._create_card(frame, "连接与关卡工具")
        r1 = tk.Frame(card2, bg=CLR_PANEL)
        r1.pack(fill=tk.X, padx=12, pady=(0, 6))
        CyberButton(r1, "Attach", self.cmd_attach, bg_color=COLOR_BLUE, fg_color="#ffffff", height=30).pack(side=tk.LEFT, padx=(0, 6))
        CyberButton(r1, "刷新状态", self.refresh_status, bg_color=CLR_CARD, fg_color=CLR_TEXT_MAIN, height=30).pack(side=tk.LEFT)

        r2 = tk.Frame(card2, bg=CLR_PANEL)
        r2.pack(fill=tk.X, padx=12, pady=(0, 12))
        tk.Label(r2, text="关卡", font=FONT_BOLD, fg=CLR_TEXT_MAIN, bg=CLR_PANEL).pack(side=tk.LEFT)
        self.cmb_area = ttk.Combobox(r2, values=[str(i) for i in range(1, 6)], width=3, state="readonly")
        self.cmb_area.set("1")
        self.cmb_area.pack(side=tk.LEFT, padx=(6, 4))
        self.cmb_sub = ttk.Combobox(r2, values=[str(i) for i in range(1, 11)], width=3, state="readonly")
        self.cmb_sub.set("1")
        self.cmb_sub.pack(side=tk.LEFT, padx=(0, 8))
        CyberButton(r2, "跳转", self.cmd_jump_custom, bg_color=CLR_CARD, fg_color=COLOR_BLUE, height=30).pack(side=tk.LEFT)

        card3 = self._create_card(frame, "自动化快进模块")
        tk.Label(
            card3,
            text="自动化推进与屏蔽流程，加速强化学习训练与调试。支持随时切换测试。",
            font=("Microsoft YaHei UI", 8),
            fg=CLR_TEXT_MUTED,
            bg=CLR_PANEL,
        ).pack(anchor="w", padx=12, pady=(0, 8))

        row1 = tk.Frame(card3, bg=CLR_PANEL)
        row1.pack(fill=tk.X, padx=12, pady=(0, 6))
        self.btn_toggle_auto_dialogue = CyberButton(
            row1,
            f"戴夫对话自动推进 (按键法): {'开启' if self._auto_dialogue_enabled else '关闭'}",
            self.toggle_auto_dialogue,
            bg_color=COLOR_GREEN if self._auto_dialogue_enabled else CLR_CARD,
            fg_color="#ffffff" if self._auto_dialogue_enabled else CLR_TEXT_MUTED,
            height=30,
        )
        self.btn_toggle_auto_dialogue.pack(fill=tk.X)

        row2 = tk.Frame(card3, bg=CLR_PANEL)
        row2.pack(fill=tk.X, padx=12, pady=(0, 6))
        self.btn_toggle_auto_tutorial = CyberButton(
            row2,
            "全关卡教学机跳过 (内存法): 关闭",
            self.toggle_auto_tutorial,
            bg_color=CLR_CARD,
            fg_color=CLR_TEXT_MUTED,
            height=30,
        )
        self.btn_toggle_auto_tutorial.pack(fill=tk.X)

        row3 = tk.Frame(card3, bg=CLR_PANEL)
        row3.pack(fill=tk.X, padx=12, pady=(0, 6))
        self.btn_toggle_no_dave = CyberButton(
            row3,
            "关卡全局屏蔽戴夫 (补丁法): 开启",
            self.toggle_no_dave,
            bg_color=COLOR_GREEN,
            fg_color="#ffffff",
            height=30,
        )
        self.btn_toggle_no_dave.pack(fill=tk.X)

        row4 = tk.Frame(card3, bg=CLR_PANEL)
        row4.pack(fill=tk.X, padx=12, pady=(0, 12))
        self.btn_toggle_auto_intro = CyberButton(
            row4,
            "开场过场自动跳过 (CancelIntro): 开启",
            self.toggle_auto_cancel_intro,
            bg_color=COLOR_GREEN,
            fg_color="#ffffff",
            height=30,
        )
        self.btn_toggle_auto_intro.pack(fill=tk.X)

        card4 = self._create_card(frame, "自动化处理脚本")
        for text in [
            "自动收集掉落物 (Auto Collect)",
            "自动砸僵尸 (Auto Whack)",
            "解除后台失焦暂停 (Background Running)",
        ]:
            tk.Label(
                card4,
                text=text,
                font=FONT_MAIN,
                fg=CLR_TEXT_MUTED,
                bg=CLR_PANEL,
            ).pack(anchor="w", padx=14, pady=3)
        tk.Label(card4, text="", bg=CLR_PANEL, height=1).pack(pady=(0, 4))

    def _create_card(self, parent, title: str) -> tk.Frame:
        card = tk.Frame(
            parent,
            bg=CLR_PANEL,
            bd=0,
            highlightbackground=CLR_BORDER,
            highlightthickness=1,
        )
        card.pack(fill=tk.X, pady=(0, 10))
        tk.Label(
            card,
            text=title,
            font=FONT_TITLE,
            fg=CLR_TEXT_MAIN,
            bg=CLR_PANEL,
        ).pack(anchor="w", padx=12, pady=(10, 6))
        return card

    def _build_right_console(self, parent):
        stats = tk.Frame(parent, bg=CLR_PANEL, height=48)
        stats.pack(fill=tk.X, pady=(0, 8))
        stats.pack_propagate(False)
        tk.Label(
            stats,
            text="运行日志",
            font=FONT_TITLE,
            fg=CLR_TEXT_MAIN,
            bg=CLR_PANEL,
        ).pack(side=tk.LEFT, padx=(14, 10), pady=10)
        self.lbl_stats = tk.Label(
            stats,
            text="测试统计: 就绪",
            font=FONT_BOLD,
            fg=CLR_TEXT_MAIN,
            bg=CLR_PANEL,
        )
        self.lbl_stats.pack(side=tk.LEFT, pady=10)

        self.progress_bar = ttk.Progressbar(parent, orient="horizontal", mode="determinate")
        self.progress_bar.pack(fill=tk.X, pady=(0, 6))

        box = tk.Frame(parent, bg="#050811", highlightbackground=CLR_BORDER, highlightthickness=1)
        box.pack(fill=tk.BOTH, expand=True)
        self.txt_console = tk.Text(
            box,
            bg="#050811",
            fg="#e2e8f0",
            insertbackground=COLOR_BLUE,
            selectbackground=CLR_CARD,
            font=FONT_MONO,
            wrap="word",
            bd=0,
            padx=12,
            pady=10,
        )
        scroll = ttk.Scrollbar(box, orient="vertical", style="Console.Vertical.TScrollbar", command=self.txt_console.yview)
        self.txt_console.configure(yscrollcommand=scroll.set)
        self.txt_console.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.txt_console.tag_config("PASS", foreground=COLOR_GREEN, font=(FONT_MONO[0], FONT_MONO[1], "bold"))
        self.txt_console.tag_config("FAIL", foreground=COLOR_RED, font=(FONT_MONO[0], FONT_MONO[1], "bold"))
        self.txt_console.tag_config("SKIP", foreground=CLR_TEXT_MUTED, font=(FONT_MONO[0], FONT_MONO[1], "bold"))
        self.txt_console.tag_config("WARN", foreground="#94a3b8", font=(FONT_MONO[0], FONT_MONO[1], "bold"))
        self.txt_console.tag_config("INFO", foreground=COLOR_BLUE)
        self.txt_console.tag_config("MUTED", foreground=CLR_TEXT_MUTED)
        self.txt_console.tag_config("WHITE", foreground=CLR_TEXT_MAIN)
        self.txt_console.tag_config("HEADER", foreground="#bc8cff", font=(FONT_MONO[0], 10, "bold"))

        tools = tk.Frame(parent, bg=CLR_BG)
        tools.pack(fill=tk.X, pady=(8, 0))
        CyberButton(tools, "复制日志", self.copy_console, bg_color=CLR_CARD, fg_color=CLR_TEXT_MAIN, height=30).pack(side=tk.LEFT, padx=(0, 8))
        CyberButton(tools, "导出报告", self.export_report, bg_color=CLR_CARD, fg_color=CLR_TEXT_MAIN, height=30).pack(side=tk.LEFT)
        self.log("INFO", "[INFO] PvZ Gym LIVE 测试工作台就绪")
        self.log("MUTED", f"DLL: {self.client.dll_path or '未找到'}")

    # --------------------------------------------------------------- LIVE rows
    def _select_live_scene(self, area: int):
        if self._live_running or getattr(self, "_deck_testing", False):
            return
        self._live_scene = int(area)
        if not ((area - 1) * 10 + 1 <= self._live_selected_level <= area * 10):
            self._live_selected_level = (area - 1) * 10 + 1
        for key, button in self._scene_buttons.items():
            if key == area:
                button.set_colors(COLOR_BLUE, "#ffffff")
            else:
                button.set_colors(CLR_CARD, CLR_TEXT_MAIN)
        self._render_live_rows()
        if hasattr(self, "live_deck_panel"):
            self.lbl_live_selected.config(text=f"已选择 {get_level_spec(self._live_selected_level).label}")
            self.live_deck_panel.refresh()

    def _select_live_level(self, level_id: int):
        if self._live_running or getattr(self, "_deck_testing", False):
            return
        if level_id == getattr(self, "_live_selected_level", None):
            return
        self._live_selected_level = int(level_id)
        spec = get_level_spec(level_id)
        self.lbl_live_selected.config(text=f"已选择 {spec.label}")
        if hasattr(self, "live_deck_panel"):
            self.live_deck_panel.refresh()
        self._update_live_row_selection()

    def _update_live_row_selection(self):
        if not hasattr(self, "_live_row_slots") or len(self._live_row_slots) != 10:
            self._render_live_rows()
            return
        for slot in self._live_row_slots:
            selected = slot["level_id"] == self._live_selected_level
            bg_color = "#1f2937" if selected else CLR_PANEL
            slot["row"].config(
                bg=bg_color,
                highlightbackground=COLOR_BLUE if selected else CLR_BORDER,
                highlightthickness=1 if selected else 0,
            )
            slot["indicator"].config(
                text="●" if selected else "○",
                fg=COLOR_BLUE if selected else CLR_TEXT_MUTED,
                bg=bg_color,
            )
            slot["level_label"].config(bg=bg_color)
            slot["status"].config(bg=bg_color)
            slot["detail"].config(bg=bg_color)

    def _render_live_rows(self):
        specs = get_scene_specs(self._live_scene)
        self._live_row_widgets.clear()

        # 固定复用 10 行控件插槽，仅在首次加载时创建一次，后续切换场景/关卡全部就地更新，彻底消除白色加载和闪烁
        if not hasattr(self, "_live_row_slots") or len(self._live_row_slots) != 10:
            for child in self.live_rows_frame.winfo_children():
                child.destroy()
            self._live_row_slots = []
            for _ in range(10):
                row = tk.Frame(
                    self.live_rows_frame,
                    bg=CLR_PANEL,
                    highlightbackground=CLR_BORDER,
                    highlightthickness=0,
                    cursor="hand2",
                )
                row.pack(fill=tk.X, pady=1)

                indicator = tk.Label(
                    row,
                    text="○",
                    width=2,
                    font=FONT_BOLD,
                    fg=CLR_TEXT_MUTED,
                    bg=CLR_PANEL,
                    cursor="hand2",
                )
                indicator.pack(side=tk.LEFT, padx=(6, 0), pady=5)

                level_label = tk.Label(
                    row,
                    text="",
                    width=8,
                    anchor="w",
                    font=FONT_BOLD,
                    fg=CLR_TEXT_MAIN,
                    bg=CLR_PANEL,
                    cursor="hand2",
                )
                level_label.pack(side=tk.LEFT, pady=6)

                status = tk.Label(
                    row,
                    text="",
                    width=14,
                    anchor="w",
                    font=FONT_BOLD,
                    fg=CLR_TEXT_MUTED,
                    bg=CLR_PANEL,
                    cursor="hand2",
                )
                status.pack(side=tk.LEFT, pady=6)

                detail = tk.Label(
                    row,
                    text="",
                    anchor="w",
                    font=FONT_MAIN,
                    fg=CLR_TEXT_MUTED,
                    bg=CLR_PANEL,
                    cursor="hand2",
                )
                detail.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=6)

                slot = {
                    "row": row,
                    "indicator": indicator,
                    "level_label": level_label,
                    "status": status,
                    "detail": detail,
                    "level_id": 0,
                }

                def _bind_slot_click(w, s=slot):
                    w.bind("<Button-1>", lambda _e: self._select_live_level(s["level_id"]))

                _bind_slot_click(row)
                _bind_slot_click(indicator)
                _bind_slot_click(level_label)
                _bind_slot_click(status)
                _bind_slot_click(detail)

                self._live_row_slots.append(slot)

        # 就地更新 10 行文本与状态数据
        for slot, spec in zip(self._live_row_slots, specs):
            slot["level_id"] = spec.level_id
            selected = spec.level_id == self._live_selected_level
            bg_color = "#1f2937" if selected else CLR_PANEL

            slot["row"].config(
                bg=bg_color,
                highlightbackground=COLOR_BLUE if selected else CLR_BORDER,
                highlightthickness=1 if selected else 0,
            )
            slot["indicator"].config(
                text="●" if selected else "○",
                fg=COLOR_BLUE if selected else CLR_TEXT_MUTED,
                bg=bg_color,
            )
            slot["level_label"].config(
                text=spec.label,
                bg=bg_color,
            )

            state = self._live_status.get(spec.level_id, {"status": "READY", "detail": ""})
            slot["status"].config(
                text=state["status"],
                fg=LIVE_STATUS_COLORS.get(state["status"], CLR_TEXT_MUTED),
                bg=bg_color,
            )
            slot["detail"].config(
                text=state["detail"],
                bg=bg_color,
            )

            self._live_row_widgets[spec.level_id] = {
                "status": slot["status"],
                "detail": slot["detail"],
            }

        selected_spec = get_level_spec(self._live_selected_level)
        self.lbl_live_selected.config(text=f"已选择 {selected_spec.label}")
        self._refresh_live_summary()

    def _set_level_status(self, level_id: int, status: str, detail: str = ""):
        level_id = int(level_id)
        self._live_status[level_id] = {"status": status, "detail": detail}
        widgets = self._live_row_widgets.get(level_id)
        if widgets:
            widgets["status"].config(
                text=status,
                fg=LIVE_STATUS_COLORS.get(status, CLR_TEXT_MUTED),
            )
            widgets["detail"].config(text=detail)
        self._refresh_live_summary()

    def _refresh_live_summary(self):
        counts: Dict[str, int] = {}
        for state in self._live_status.values():
            counts[state["status"]] = counts.get(state["status"], 0) + 1
        self.lbl_live_summary.config(
            text=f"PASS {counts.get('PASS', 0)} · FAIL {counts.get('FAIL', 0)} · READY {counts.get('READY', 0)}"
        )

    def _detect_playthrough(self) -> str:
        state = self.client.get_runtime_state()
        if not isinstance(state, dict) or state.get("error"):
            raise RuntimeError("无法读取 runtime_state")
        finished = int(state.get("finished_adventure", -1))
        if finished < 0:
            raise RuntimeError("finished_adventure 不可用，请重新编译 pvz_gym_native.dll")
        return PLAYTHROUGH_1 if finished == 0 else PLAYTHROUGH_2

    def _set_live_controls_running(self, running: bool, paused: bool = False):
        self.btn_live_e2e.set_disabled(running)
        self.btn_live_x5.set_disabled(running)
        self.btn_live_all.set_disabled(running)
        self.btn_live_pause.set_disabled((not running) or paused)
        self.btn_live_continue.set_disabled((not running) or (not paused))
        self.btn_live_stop.set_disabled(not running)
        if hasattr(self, "cmb_live_speed"):
            self.cmb_live_speed.config(state="disabled" if running else "readonly")
        if hasattr(self, "live_deck_panel"):
            self.live_deck_panel.set_busy(running)
            self.cmb_live_playthrough.config(state="disabled" if running else "readonly")

    def _selected_live_speed_config(self) -> Dict[str, object]:
        label = self.var_live_speed.get()
        config = LIVE_SPEED_CONFIGS.get(label)
        if config is None:
            raise ValueError(f"unknown LIVE simulation rate: {label}")
        return dict(config)

    def _begin_live(self, title: str) -> bool:
        if getattr(self, "_deck_testing", False):
            self.log("WARN", "[LIVE] 选卡测试正在运行")
            return False
        if self._live_running:
            self.log("WARN", "[LIVE] 已有测试正在运行")
            return False
        try:
            mode, cards = self.live_deck_panel.settings()
        except ValueError as exc:
            self.live_deck_panel.show_result(str(exc), True)
            return False
        self._active_live_seed_priority = cards if mode == "custom" else None
        self._active_live_seed_selection_mode = mode
        try:
            self._active_live_seed_profiles = self.live_deck_panel.selection_profiles()
        except ValueError as exc:
            self.live_deck_panel.show_result(str(exc), True)
            return False
        self._active_live_playthrough = (PLAYTHROUGH_1 if self.var_live_deck_playthrough.get() == "1"
                                         else PLAYTHROUGH_2)
        self._live_running = True
        self._live_pause_event = threading.Event()
        self._live_pause_event.set()
        self._live_stop_event = threading.Event()
        self._active_live_speed = self._selected_live_speed_config()
        self._set_live_controls_running(True, False)
        self.lbl_live_current.config(text=title, fg=COLOR_BLUE)
        self.log(
            "INFO",
            f"[LIVE] simulation rate: {self.var_live_speed.get()}",
        )
        return True

    def _finish_live(self, text: str, failed: bool = False):
        self._live_running = False
        self._live_current_level = None
        self._live_pause_event = None
        self._live_stop_event = None
        self._set_live_controls_running(False)
        self.lbl_live_current.config(
            text=text,
            fg=COLOR_RED if failed else COLOR_GREEN,
        )
        self.refresh_status()

    def _progress_callback(self, spec):
        def progress(phase: str, message: str):
            def update():
                self.lbl_live_current.config(
                    text=f"{spec.label} · {phase} · {message[:70]}",
                    fg=COLOR_BLUE,
                )
                self.log("INFO", f"[{spec.label}:{phase}] {message}")
            self.root.after(0, update)
        return progress

    def _start_selected_live(self, episodes: int):
        spec = get_level_spec(self._live_selected_level)
        if not self._begin_live(f"{spec.label} · {'E2E' if episodes == 1 else 'E2E x5'}"):
            return
        threading.Thread(
            target=self._selected_live_worker,
            args=(spec, episodes),
            daemon=True,
        ).start()

    def _selected_live_worker(self, spec, episodes: int):
        try:
            playthrough = self._active_live_playthrough
            self._live_current_level = spec.level_id
            self.root.after(0, self._set_level_status, spec.level_id, "RUNNING", "")
            progress = self._progress_callback(spec)
            if episodes == 1:
                result = run_early_level_e2e(
                    self.client,
                    spec,
                    expected_path=playthrough,
                    seed_priority=self._active_live_seed_priority,
                    seed_selection_mode=self._active_live_seed_selection_mode,
                    progress=progress,
                    **self._active_live_speed,
                    pause_event=self._live_pause_event,
                    stop_event=self._live_stop_event,
                )
                if result.success:
                    detail = result.outcome
                    self.root.after(0, self._set_level_status, spec.level_id, "PASS", detail)
                    self.root.after(0, self.log, "PASS", f"[{spec.label}] PASS · {detail}")
                    self.root.after(0, self._finish_live, f"{spec.label} · PASS", False)
                else:
                    status = "STOPPED" if result.outcome == "ABORTED" else "FAIL"
                    on, _ = self.client.is_game_on()
                    if not on:
                        status = "STOPPED"
                        self._stop_live()
                    self.root.after(0, self._set_level_status, spec.level_id, status, result.outcome)
                    self.root.after(0, self.log, "FAIL" if status == "FAIL" else "WARN", f"[{spec.label}] {status} · {result.outcome}: {result.message}")
                    self.root.after(0, self._finish_live, f"{spec.label} · {status}", status == "FAIL")
            else:
                result = run_early_level_stress(
                    self.client,
                    spec,
                    episodes=episodes,
                    expected_path=playthrough,
                    seed_priority=self._active_live_seed_priority,
                    seed_selection_mode=self._active_live_seed_selection_mode,
                    progress=progress,
                    **self._active_live_speed,
                    pause_event=self._live_pause_event,
                    stop_event=self._live_stop_event,
                )
                if result.success:
                    detail = f"{result.wins}W/{result.losses}L"
                    self.root.after(0, self._set_level_status, spec.level_id, "PASS", detail)
                    self.root.after(0, self.log, "PASS", f"[{spec.label}] x5 PASS · {detail}")
                    self.root.after(0, self._finish_live, f"{spec.label} · x5 PASS", False)
                else:
                    aborted = any(r.outcome == "ABORTED" for r in result.episode_results)
                    on, _ = self.client.is_game_on()
                    if not on:
                        aborted = True
                        self._stop_live()
                    status = "STOPPED" if aborted else "FAIL"
                    detail = f"{result.completed_episodes}/{result.requested_episodes}"
                    self.root.after(0, self._set_level_status, spec.level_id, status, detail)
                    self.root.after(0, self.log, "FAIL" if status == "FAIL" else "WARN", f"[{spec.label}] x5 {status} · {detail}")
                    self.root.after(0, self._finish_live, f"{spec.label} · x5 {status}", status == "FAIL")
        except Exception as exc:
            self.root.after(0, self._set_level_status, spec.level_id, "FAIL", str(exc))
            self.root.after(0, self.log, "FAIL", f"[{spec.label}] LIVE exception: {exc}")
            self.root.after(0, self._finish_live, f"{spec.label} · FAIL", True)

    def _start_all_live(self):
        if not self._begin_live("ALL 50 · RUNNING"):
            return
        threading.Thread(target=self._all_live_worker, daemon=True).start()

    def _all_live_worker(self):
        passed = failed = 0
        try:
            playthrough = self._active_live_playthrough
            specs = all_level_specs()
            for index, spec in enumerate(specs, start=1):
                if self._live_stop_event is not None and self._live_stop_event.is_set():
                    break
                while self._live_pause_event is not None and not self._live_pause_event.is_set():
                    if self._live_stop_event is not None and self._live_stop_event.is_set():
                        break
                    time.sleep(0.05)
                if self._live_stop_event is not None and self._live_stop_event.is_set():
                    break

                self._live_current_level = spec.level_id
                self.root.after(0, self._set_level_status, spec.level_id, "RUNNING", f"{index}/50")
                self.root.after(0, self.lbl_live_current.config, {"text": f"ALL 50 · {index}/50 · {spec.label}", "fg": COLOR_BLUE})
                mode, cards = self._active_live_seed_profiles.get(
                    (spec.level_id, 1 if playthrough == PLAYTHROUGH_1 else 2), ("fixed", None))
                result = run_early_level_e2e(
                    self.client,
                    spec,
                    expected_path=playthrough,
                    seed_priority=cards,
                    seed_selection_mode=mode,
                    progress=self._progress_callback(spec),
                    **self._active_live_speed,
                    pause_event=self._live_pause_event,
                    stop_event=self._live_stop_event,
                )
                if result.success:
                    passed += 1
                    self.root.after(0, self._set_level_status, spec.level_id, "PASS", result.outcome)
                else:
                    if result.outcome == "ABORTED":
                        self.root.after(0, self._set_level_status, spec.level_id, "STOPPED", "ABORTED")
                        break
                    on, _ = self.client.is_game_on()
                    if not on:
                        self.root.after(0, self._set_level_status, spec.level_id, "STOPPED", "ABORTED")
                        self._stop_live()
                        break
                    failed += 1
                    self.root.after(0, self._set_level_status, spec.level_id, "FAIL", result.outcome)
                    self.root.after(0, self.log, "FAIL", f"[{spec.label}] {result.outcome}: {result.message}")

            stopped = self._live_stop_event is not None and self._live_stop_event.is_set()
            if stopped:
                self.root.after(0, self._finish_live, f"ALL 50 · STOPPED · {passed} PASS / {failed} FAIL", failed > 0)
            else:
                self.root.after(0, self.log, "PASS" if failed == 0 else "FAIL", f"[ALL 50] 完成: PASS={passed}, FAIL={failed}")
                self.root.after(0, self._finish_live, f"ALL 50 · {passed} PASS / {failed} FAIL", failed > 0)
        except Exception as exc:
            self.root.after(0, self.log, "FAIL", f"[ALL 50] exception: {exc}")
            self.root.after(0, self._finish_live, "ALL 50 · FAIL", True)

    def _pause_live(self):
        if not self._live_running or self._live_pause_event is None:
            return
        self._live_pause_event.clear()
        if self._live_current_level is not None:
            current = self._live_status[self._live_current_level]
            self._set_level_status(self._live_current_level, "PAUSED", current.get("detail", ""))
        self.lbl_live_current.config(text="PAUSED", fg=LIVE_STATUS_COLORS["PAUSED"])
        self._set_live_controls_running(True, True)

    def _continue_live(self):
        if not self._live_running or self._live_pause_event is None:
            return
        self._live_pause_event.set()
        if self._live_current_level is not None:
            current = self._live_status[self._live_current_level]
            self._set_level_status(self._live_current_level, "RUNNING", current.get("detail", ""))
        self.lbl_live_current.config(text="RUNNING", fg=COLOR_BLUE)
        self._set_live_controls_running(True, False)

    def _stop_live(self):
        if not self._live_running or self._live_stop_event is None:
            return
        self._live_stop_event.set()
        if self._live_pause_event is not None:
            self._live_pause_event.set()
        self.lbl_live_current.config(text="STOPPING...", fg=CLR_TEXT_MUTED)

    # ---------------------------------------------------------- console/status
    def log(self, tag: str, text: str):
        if not hasattr(self, "txt_console"):
            return
        self.txt_console.insert(tk.END, str(text) + "\n", tag)
        self.txt_console.see(tk.END)
        logger.debug("[%s] %s", tag, text)

    def clear_console(self):
        self.txt_console.delete("1.0", tk.END)

    def copy_console(self):
        text = self.txt_console.get("1.0", tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        messagebox.showinfo("提示", "日志已复制到剪贴板")

    def export_report(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".log",
            filetypes=[("Log Files", "*.log"), ("Text Files", "*.txt")],
            initialfile=f"pvz_live_report_{time.strftime('%Y%m%d_%H%M%S')}.log",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self.txt_console.get("1.0", tk.END))
        except Exception as exc:
            messagebox.showerror("错误", f"无法写入文件: {exc}")

    def refresh_status(self):
        if self.client.is_loaded:
            self.badge_dll.config(text="DLL: 已加载", bg=COLOR_GREEN, fg="#ffffff")
        else:
            self.badge_dll.config(text="DLL: 未加载", bg=COLOR_RED, fg="#ffffff")

        self._sync_fast_forward_state()

        on, _ = self.client.is_game_on()
        if not on:
            self.badge_game.config(text="进程: 离线", bg=CLR_CARD, fg=CLR_TEXT_MUTED)
            self.badge_ui.config(text="阶段: --", bg=CLR_CARD, fg=CLR_TEXT_MUTED)
            self.badge_level.config(text="关卡: --", bg=CLR_CARD, fg=CLR_TEXT_MUTED)
            self.badge_clock.config(text="时钟: 0", bg=CLR_CARD, fg=CLR_TEXT_MUTED)
            self.badge_paused.config(text="暂停: --", bg=CLR_CARD, fg=CLR_TEXT_MUTED)
            return

        self.badge_game.config(text="进程: 在线", bg=COLOR_GREEN, fg="#ffffff")
        ui, ui_desc = self.client.get_ui()
        self.badge_ui.config(text=f"UI: {ui_desc}", bg=COLOR_BLUE, fg=CLR_TEXT_DARK)
        current = self.client.get_current_level()
        if current.get("success"):
            self.badge_level.config(text=f"关卡: {current.get('level_str')}", bg=CLR_CARD, fg=COLOR_BLUE)
        if ui == 3:
            clock = self.client.get_game_clock()
            paused = self.client.is_paused()
            self.badge_clock.config(text=f"时钟: {clock}", bg=CLR_CARD, fg=COLOR_BLUE)
            self.badge_paused.config(
                text="暂停: 是" if paused else "暂停: 否",
                bg=CLR_CARD,
                fg="#94a3b8" if paused else COLOR_GREEN,
            )
        else:
            self.badge_clock.config(text="时钟: 0", bg=CLR_CARD, fg=CLR_TEXT_MUTED)
            self.badge_paused.config(text="暂停: --", bg=CLR_CARD, fg=CLR_TEXT_MUTED)

    # -------------------------------------------------------------- basic tools
    def toggle_auto_dialogue(self):
        new_val = not self._auto_dialogue_enabled
        if not self.client.set_auto_dialogue(new_val):
            self._sync_fast_forward_state()
            return
        self._auto_dialogue_enabled = self.client.get_auto_dialogue()
        self._update_fast_forward_ui()
        self.log("INFO", f"[*] 戴夫对话自动推进 (按键法): {'开启' if self._auto_dialogue_enabled else '关闭'}")

    def toggle_auto_tutorial(self):
        new_val = not self._auto_tutorial_enabled
        self.client.set_auto_skip_tutorial(new_val)
        self._auto_tutorial_enabled = new_val
        self._update_fast_forward_ui()
        self.log("INFO", f"[*] 全关卡教学机跳过 (内存法): {'开启' if new_val else '关闭'}")

    def toggle_no_dave(self):
        new_val = not self._no_dave_enabled
        self.client.set_no_crazy_dave(new_val)
        self._no_dave_enabled = new_val
        self._update_fast_forward_ui()
        self.log("INFO", f"[*] 关卡全局屏蔽戴夫 (补丁法): {'开启' if new_val else '关闭'}")

    def toggle_auto_cancel_intro(self):
        new_val = not self._auto_cancel_intro_enabled
        self.client.set_auto_cancel_intro(new_val)
        self._auto_cancel_intro_enabled = new_val
        self._update_fast_forward_ui()
        self.log("INFO", f"[*] 开场过场自动跳过 (CancelIntro): {'开启' if new_val else '关闭'}")

    def _update_fast_forward_ui(self):
        if getattr(self, "btn_toggle_auto_dialogue", None):
            if self._auto_dialogue_enabled:
                self.btn_toggle_auto_dialogue.set_text("戴夫对话自动推进 (按键法): 开启")
                self.btn_toggle_auto_dialogue.set_colors(COLOR_GREEN, "#ffffff")
            else:
                self.btn_toggle_auto_dialogue.set_text("戴夫对话自动推进 (按键法): 关闭")
                self.btn_toggle_auto_dialogue.set_colors(CLR_CARD, CLR_TEXT_MUTED)

        if getattr(self, "btn_toggle_auto_tutorial", None):
            if self._auto_tutorial_enabled:
                self.btn_toggle_auto_tutorial.set_text("全关卡教学机跳过 (内存法): 开启")
                self.btn_toggle_auto_tutorial.set_colors(COLOR_GREEN, "#ffffff")
            else:
                self.btn_toggle_auto_tutorial.set_text("全关卡教学机跳过 (内存法): 关闭")
                self.btn_toggle_auto_tutorial.set_colors(CLR_CARD, CLR_TEXT_MUTED)

        if getattr(self, "btn_toggle_no_dave", None):
            if self._no_dave_enabled:
                self.btn_toggle_no_dave.set_text("关卡全局屏蔽戴夫 (补丁法): 开启")
                self.btn_toggle_no_dave.set_colors(COLOR_GREEN, "#ffffff")
            else:
                self.btn_toggle_no_dave.set_text("关卡全局屏蔽戴夫 (补丁法): 关闭")
                self.btn_toggle_no_dave.set_colors(CLR_CARD, CLR_TEXT_MUTED)

        if getattr(self, "btn_toggle_auto_intro", None):
            if self._auto_cancel_intro_enabled:
                self.btn_toggle_auto_intro.set_text("开场过场自动跳过 (CancelIntro): 开启")
                self.btn_toggle_auto_intro.set_colors(COLOR_GREEN, "#ffffff")
            else:
                self.btn_toggle_auto_intro.set_text("开场过场自动跳过 (CancelIntro): 关闭")
                self.btn_toggle_auto_intro.set_colors(CLR_CARD, CLR_TEXT_MUTED)

    def _sync_fast_forward_state(self):
        if hasattr(self, "client") and self.client and self.client.is_loaded:
            try:
                self._auto_dialogue_enabled = self.client.get_auto_dialogue()
                self._auto_tutorial_enabled = self.client.get_auto_skip_tutorial()
                self._no_dave_enabled = self.client.get_no_crazy_dave()
                self._auto_cancel_intro_enabled = self.client.get_auto_cancel_intro()
                self.client.set_auto_whack(True)
                self._auto_whack_enabled = True
            except Exception:
                pass
        self._update_fast_forward_ui()

    def toggle_game_topmost(self):
        self._is_topmost = not getattr(self, "_is_topmost", False)
        self._update_topmost_ui()
        if hasattr(self, "client") and self.client and hasattr(self.client, "set_window_always_on_top"):
            self.client.set_window_always_on_top(self._is_topmost)
        status_text = "开启" if self._is_topmost else "关闭"
        self.log("INFO", f"[*] 游戏窗口置顶已{status_text}")

    def _update_topmost_ui(self):
        if not hasattr(self, "btn_topmost") or self.btn_topmost is None:
            return
        if getattr(self, "_is_topmost", False):
            self.btn_topmost.set_text("游戏置顶: 开")
            self.btn_topmost.set_colors(COLOR_BLUE, "#ffffff")
        else:
            self.btn_topmost.set_text("游戏置顶: 关")
            self.btn_topmost.set_colors(CLR_CARD, CLR_TEXT_MUTED)

    def cmd_attach(self):
        ok, message = self.client.attach()
        self.log("PASS" if ok else "FAIL", f"Attach: {message}")
        self.refresh_status()

    def cmd_jump_custom(self):
        try:
            area = int(self.cmb_area.get())
            sub = int(self.cmb_sub.get())
        except Exception:
            return
        code, message = self.client.select_level(area, sub)
        self.log("PASS" if code == 0 else "FAIL", f"select_level({area}-{sub}): {code} {message}")
        self.refresh_status()

    # --------------------------------------------------------------- daemon
    def _start_daemon_loop(self):
        self._last_attach_time = 0.0
        self._daemon_step()

    def _daemon_step(self):
        if getattr(self, "_window_closing", False) or not getattr(self, "root", None) or not self.root.winfo_exists():
            return
        try:
            if self.client.is_loaded:
                on, _ = self.client.is_game_on()
                if not on and time.time() - self._last_attach_time >= 1.0:
                    self._last_attach_time = time.time()
                    self.client.attach()
                if on and not getattr(self, "_game_was_on", False):
                    self._game_was_on = True
                    self.log("PASS", "[自动连接] 检测到游戏进程上线，已自动完成补丁 Attach")
                elif not on and getattr(self, "_game_was_on", False):
                    self._game_was_on = False
                    self.log("MUTED", "[状态变更] 游戏进程已退出/离线")
                    if getattr(self, "_live_running", False):
                        self.log("WARN", "[自动终止] 检测到游戏进程异常退出，已自动触发测试终止")
                        self._stop_live()
                    if getattr(self, "_static_test_running", False):
                        self._static_test_stop = True
                self.refresh_status()
        except Exception as exc:
            logger.debug("daemon: %s", exc)
        self.root.after(500, self._daemon_step)

    # ----------------------------------------------------------- static tests
    def run_all_tests_async(self):
        if self._static_test_running:
            return
        self._static_test_running = True
        self._static_test_stop = False
        self.btn_run_all.set_disabled(True)
        threading.Thread(target=self._run_all_worker, daemon=True).start()

    def _run_all_worker(self):
        tests = self.suite.get_all_tests()
        total = len(tests)
        passed = failed = skipped = 0
        start = time.perf_counter()
        self.root.after(0, self.progress_bar.configure, {"value": 0, "maximum": total})
        try:
            for idx, test in enumerate(tests, start=1):
                if getattr(self, "_static_test_stop", False):
                    break
                result = self.suite.run_single(test)
                if result.status == "PASS":
                    passed += 1
                elif result.status == "FAIL":
                    failed += 1
                else:
                    skipped += 1
                self.root.after(0, self._append_case_result, idx, total, result)
                self.root.after(0, self.progress_bar.configure, {"value": idx})
        finally:
            self._static_test_running = False
            elapsed = (time.perf_counter() - start) * 1000.0
            self.root.after(0, self._finish_static_tests, total, passed, failed, skipped, elapsed)

    def _append_case_result(self, idx: int, total: int, result: TestCaseResult):
        self.log("WHITE", f"[{idx:02d}/{total:02d}] {result.case_id} · {result.name}")
        self.log(result.status, f"  {result.status} · {result.elapsed_ms:.2f}ms · {result.message}")

    def _finish_static_tests(self, total, passed, failed, skipped, elapsed):
        self.btn_run_all.set_disabled(False)
        self.lbl_stats.config(
            text=f"C++ 测试: {total} | PASS {passed} | FAIL {failed} | SKIP {skipped} | {elapsed:.1f}ms",
            fg=COLOR_GREEN if failed == 0 else COLOR_RED,
        )


def request_workbench_close(app):
    """Keep Tk alive while the trainer saves and releases its owned workers."""
    if getattr(app, "_window_closing", False):
        return
    app._window_closing = True
    if getattr(app, "_training_running", False):
        try:
            app.log("WARN", "[SYSTEM] 正在关闭软件：检测到训练仍在运行，正在安全保存进度并关闭游戏进程，请稍候...")
            if hasattr(app, "lbl_training_progress"):
                app.lbl_training_progress.config(
                    text="正在退出：等待多进程环境退出并保存最新断点...",
                    fg=COLOR_GOLD,
                )
        except Exception:
            pass
        app._stop_training()

    def finish_when_idle():
        if getattr(app, "_training_running", False):
            app.root.after(100, finish_when_idle)
            return
        try:
            if hasattr(app, "client") and app.client:
                app.client.safe_restore_game()
        except Exception:
            pass
        try:
            from .q_heatmap_overlay import get_q_heatmap_overlay
            get_q_heatmap_overlay().stop()
        except Exception:
            pass
        app.root.destroy()

    finish_when_idle()


def launch_gui():
    root = tk.Tk()
    app = PvzGymUnitTestGUI(root)

    root.protocol("WM_DELETE_WINDOW", lambda: request_workbench_close(app))
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
