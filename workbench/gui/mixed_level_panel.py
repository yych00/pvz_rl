"""Mixed level and deck selection panel for Gym training."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Sequence

try:
    from train.curriculum import (
        SCENE_NAMES,
        FUN_SUB,
        CONVEYOR_SUB,
        generate_mixed_level_pool,
        format_level_pool_summary,
    )
except ImportError:
    from train.curriculum import (
        SCENE_NAMES,
        FUN_SUB,
        CONVEYOR_SUB,
        generate_mixed_level_pool,
        format_level_pool_summary,
    )


class MixedLevelPanel(tk.Frame):
    """支持5大场景多选、周目多选、趣味关/传送带过滤及选卡联动的混合配置面板。"""

    def __init__(
        self,
        parent: tk.Widget,
        ui: any,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent, bg=ui.CLR_CARD)
        self.ui = ui
        self._on_change = on_change

        # 模式变量: single (单一关卡) 或 mixed (混合关卡/选卡)
        self.var_mode = tk.StringVar(value="single")

        # 5个大场景多选变量 (1..5)
        self.scene_vars: dict[int, tk.BooleanVar] = {
            1: tk.BooleanVar(value=True),   # 默认勾选白天
            2: tk.BooleanVar(value=False),
            3: tk.BooleanVar(value=False),
            4: tk.BooleanVar(value=False),
            5: tk.BooleanVar(value=False),
        }

        # 周目变量 (1, 2)
        self.playthrough_vars: dict[int, tk.BooleanVar] = {
            1: tk.BooleanVar(value=True),   # 默认勾选一周目
            2: tk.BooleanVar(value=False),
        }

        # 特殊关卡过滤变量 (默认均不勾选)
        self.var_include_fun = tk.BooleanVar(value=False)        # x-5
        self.var_include_conveyor = tk.BooleanVar(value=False)   # x-10

        # 选卡配合策略
        self.var_deck_strategy = tk.StringVar(value="每关预设卡组")
        self.var_shuffle_slots = tk.BooleanVar(value=True)      # 默认开启卡槽打乱

        self._explicit_level_pool: tuple[tuple[int, int], ...] | None = None
        self._interactive_widgets: list[tk.Widget] = []

        self._build_ui()

    def _build_ui(self) -> None:
        ui = self.ui

        # 1. 顶部模式切换栏
        mode_bar = tk.Frame(self, bg=ui.CLR_CARD)
        mode_bar.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            mode_bar, text="模式选择:", font=("Microsoft YaHei UI", 9, "bold"),
            fg=ui.CLR_TEXT_MUTED, bg=ui.CLR_CARD,
        ).pack(side=tk.LEFT, padx=(0, 10))

        rb_single = tk.Radiobutton(
            mode_bar, text="单一关卡模式", variable=self.var_mode, value="single",
            bg=ui.CLR_CARD, fg=ui.CLR_TEXT_MAIN, selectcolor="#1c2d3d",
            activebackground=ui.CLR_CARD, activeforeground=ui.COLOR_BLUE,
            command=self._on_mode_toggled, font=ui.FONT_MAIN,
        )
        rb_single.pack(side=tk.LEFT, padx=(0, 15))

        rb_mixed = tk.Radiobutton(
            mode_bar, text="混合关卡 / 选卡模式", variable=self.var_mode, value="mixed",
            bg=ui.CLR_CARD, fg=ui.COLOR_BLUE, selectcolor="#1c2d3d",
            activebackground=ui.CLR_CARD, activeforeground=ui.COLOR_BLUE,
            command=self._on_mode_toggled, font=ui.FONT_BOLD,
        )
        rb_mixed.pack(side=tk.LEFT)

        self._interactive_widgets = [
            rb_single, rb_mixed,
        ]

        # 2. 混合关卡配置容器 (当为 mixed 时展开)
        self.mixed_container = tk.Frame(
            self, bg=ui.CLR_PANEL, highlightbackground=ui.CLR_BORDER,
            highlightthickness=1, padx=10, pady=8,
        )

        # 2.1 场景多选行 (5个大关多选勾选框 + 全选/清空)
        scene_row = tk.Frame(self.mixed_container, bg=ui.CLR_PANEL)
        scene_row.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            scene_row, text="场景多选:", font=("Microsoft YaHei UI", 8, "bold"),
            fg=ui.CLR_TEXT_MUTED, bg=ui.CLR_PANEL,
        ).pack(side=tk.LEFT, padx=(0, 8))

        scene_labels = {
            1: "白天 (1-x)",
            2: "黑夜 (2-x)",
            3: "泳池 (3-x)",
            4: "浓雾 (4-x)",
            5: "屋顶 (5-x)",
        }
        for s_id in range(1, 6):
            cb = tk.Checkbutton(
                scene_row, text=scene_labels[s_id], variable=self.scene_vars[s_id],
                bg=ui.CLR_PANEL, fg=ui.CLR_TEXT_MAIN, selectcolor="#111a25",
                activebackground=ui.CLR_PANEL, activeforeground=ui.COLOR_BLUE,
                font=ui.FONT_MAIN, command=self._handle_change,
            )
            cb.pack(side=tk.LEFT, padx=(0, 8))
            self._interactive_widgets.append(cb)

        btn_all = tk.Button(
            scene_row, text="全选", bg="#1c2d3d", fg=ui.COLOR_BLUE,
            relief="flat", font=("Microsoft YaHei UI", 8),
            command=self._select_all_scenes, padx=6, pady=1,
        )
        btn_all.pack(side=tk.LEFT, padx=(4, 2))
        self._interactive_widgets.append(btn_all)

        btn_none = tk.Button(
            scene_row, text="清空", bg="#1c2d3d", fg=ui.CLR_TEXT_MUTED,
            relief="flat", font=("Microsoft YaHei UI", 8),
            command=self._clear_all_scenes, padx=6, pady=1,
        )
        btn_none.pack(side=tk.LEFT, padx=2)
        self._interactive_widgets.append(btn_none)

        # 2.2 规则过滤行 (周目多选 + 特殊关卡多选)
        rule_row = tk.Frame(self.mixed_container, bg=ui.CLR_PANEL)
        rule_row.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            rule_row, text="周目多选:", font=("Microsoft YaHei UI", 8, "bold"),
            fg=ui.CLR_TEXT_MUTED, bg=ui.CLR_PANEL,
        ).pack(side=tk.LEFT, padx=(0, 8))

        cb_p1 = tk.Checkbutton(
            rule_row, text="一周目 (默认)", variable=self.playthrough_vars[1],
            bg=ui.CLR_PANEL, fg=ui.CLR_TEXT_MAIN, selectcolor="#111a25",
            activebackground=ui.CLR_PANEL, activeforeground=ui.COLOR_BLUE,
            font=ui.FONT_MAIN, command=self._handle_change,
        )
        cb_p1.pack(side=tk.LEFT, padx=(0, 10))
        self._interactive_widgets.append(cb_p1)

        cb_p2 = tk.Checkbutton(
            rule_row, text="二周目", variable=self.playthrough_vars[2],
            bg=ui.CLR_PANEL, fg=ui.CLR_TEXT_MAIN, selectcolor="#111a25",
            activebackground=ui.CLR_PANEL, activeforeground=ui.COLOR_BLUE,
            font=ui.FONT_MAIN, command=self._handle_change,
        )
        cb_p2.pack(side=tk.LEFT, padx=(0, 18))
        self._interactive_widgets.append(cb_p2)

        tk.Label(
            rule_row, text="| 特殊关卡:", font=("Microsoft YaHei UI", 8, "bold"),
            fg=ui.CLR_TEXT_MUTED, bg=ui.CLR_PANEL,
        ).pack(side=tk.LEFT, padx=(0, 8))

        cb_fun = tk.Checkbutton(
            rule_row, text="趣味关卡 (x-5)", variable=self.var_include_fun,
            bg=ui.CLR_PANEL, fg=ui.CLR_TEXT_MAIN, selectcolor="#111a25",
            activebackground=ui.CLR_PANEL, activeforeground=ui.COLOR_BLUE,
            font=ui.FONT_MAIN, command=self._handle_change,
        )
        cb_fun.pack(side=tk.LEFT, padx=(0, 10))
        self._interactive_widgets.append(cb_fun)

        cb_conv = tk.Checkbutton(
            rule_row, text="传送带关卡 (x-10)", variable=self.var_include_conveyor,
            bg=ui.CLR_PANEL, fg=ui.CLR_TEXT_MAIN, selectcolor="#111a25",
            activebackground=ui.CLR_PANEL, activeforeground=ui.COLOR_BLUE,
            font=ui.FONT_MAIN, command=self._handle_change,
        )
        cb_conv.pack(side=tk.LEFT)
        self._interactive_widgets.append(cb_conv)

        # 2.3 选卡配合与增强策略行
        deck_row = tk.Frame(self.mixed_container, bg=ui.CLR_PANEL)
        deck_row.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            deck_row, text="选卡配合:", font=("Microsoft YaHei UI", 8, "bold"),
            fg=ui.CLR_TEXT_MUTED, bg=ui.CLR_PANEL,
        ).pack(side=tk.LEFT, padx=(0, 8))

        cmb_strategy = ttk.Combobox(
            deck_row, textvariable=self.var_deck_strategy,
            values=("每关预设卡组", "自动针对僵尸"),
            state="readonly", width=14,
        )
        cmb_strategy.pack(side=tk.LEFT, padx=(0, 14))
        cmb_strategy.bind("<<ComboboxSelected>>", lambda _: self._handle_change())
        self._interactive_widgets.append(cmb_strategy)

        cb_shuffle = tk.Checkbutton(
            deck_row, text="卡槽顺序随机打乱 (防止模型死记卡槽编号，增强泛化)",
            variable=self.var_shuffle_slots,
            bg=ui.CLR_PANEL, fg=ui.COLOR_GREEN, selectcolor="#111a25",
            activebackground=ui.CLR_PANEL, activeforeground=ui.COLOR_GREEN,
            font=ui.FONT_MAIN, command=self._handle_change,
        )
        cb_shuffle.pack(side=tk.LEFT)
        self._interactive_widgets.append(cb_shuffle)

        # 2.4 实时关卡池统计提示卡
        summary_frame = tk.Frame(
            self.mixed_container, bg="#111a25", padx=8, pady=4,
            highlightbackground=ui.CLR_BORDER, highlightthickness=1,
        )
        summary_frame.pack(fill=tk.X, pady=(2, 0))

        self.lbl_pool_summary = tk.Label(
            summary_frame, text="", bg="#111a25", fg=ui.COLOR_BLUE,
            font=("Microsoft YaHei UI", 8, "bold"), anchor="w",
        )
        self.lbl_pool_summary.pack(side=tk.LEFT)

        self._update_pool_summary()

    def _on_mode_toggled(self) -> None:
        if self.is_mixed_mode():
            self.mixed_container.pack(fill=tk.X, pady=(0, 6))
        else:
            self.mixed_container.pack_forget()
        self._handle_change()

    def _select_all_scenes(self) -> None:
        self._explicit_level_pool = None
        for var in self.scene_vars.values():
            var.set(True)
        self._handle_change()

    def _clear_all_scenes(self) -> None:
        self._explicit_level_pool = None
        for var in self.scene_vars.values():
            var.set(False)
        self._handle_change()

    def _handle_change(self) -> None:
        # 防空保护: 至少选一个周目
        if not any(v.get() for v in self.playthrough_vars.values()):
            self.playthrough_vars[1].set(True)
        self._update_pool_summary()
        if self._on_change:
            self._on_change()

    def _update_pool_summary(self) -> None:
        pool = self.get_level_pool()
        if pool is None:
            self.lbl_pool_summary.config(
                text="当前模式: 单一关卡模式 (由下方单一关卡与周目选择器控制)",
                fg=self.ui.CLR_TEXT_MUTED,
            )
        elif not pool:
            self.lbl_pool_summary.config(
                text="⚠️ 未选中任何场景！请至少勾选一个场景。",
                fg="#f87171",
            )
        else:
            summary = format_level_pool_summary(pool)
            self.lbl_pool_summary.config(
                text=f"当前混合关卡池: {summary} | 抽选: 随机均匀 (Uniform)",
                fg=self.ui.COLOR_BLUE,
            )

    def is_mixed_mode(self) -> bool:
        return self.var_mode.get() == "mixed"

    def get_level_pool(self) -> tuple[tuple[int, int], ...] | None:
        """获取当前配置生成的关卡池。若是单一模式则返回 None。"""
        if not self.is_mixed_mode():
            return None
        if self._explicit_level_pool is not None:
            return self._explicit_level_pool
        scenes = [s for s, v in self.scene_vars.items() if v.get()]
        playthroughs = [p for p, v in self.playthrough_vars.items() if v.get()]
        return generate_mixed_level_pool(
            scenes=scenes,
            playthroughs=playthroughs,
            include_fun=self.var_include_fun.get(),
            include_conveyor=self.var_include_conveyor.get(),
        )

    def get_seed_selection_mode(self) -> str:
        """根据策略与打乱开关返回底层 seed_selection_mode"""
        strategy = self.var_deck_strategy.get()
        if strategy == "自动针对僵尸":
            return "adaptive_shuffled" if self.var_shuffle_slots.get() else "adaptive"
        # 每关预设卡组
        if self.var_shuffle_slots.get():
            return "fixed_shuffled"
        return "fixed"

    def set_from_level_pool(
        self,
        level_pool: Sequence[tuple[int, int]] | None,
        seed_selection_mode: str | None = None,
    ) -> None:
        """从已有的关卡池和选卡模式恢复面板状态"""
        if not level_pool:
            self._explicit_level_pool = None
            self.var_mode.set("single")
            self.mixed_container.pack_forget()
            self._handle_change()
            return

        normalized = tuple((int(lvl), int(p)) for lvl, p in level_pool)
        self.var_mode.set("mixed")

        # 还原大关场景
        scenes_in_pool = {(lvl - 1) // 10 + 1 for lvl, _ in normalized}
        for s_id, var in self.scene_vars.items():
            var.set(s_id in scenes_in_pool)

        # 还原周目
        playthroughs_in_pool = {p for _, p in normalized}
        for p_id, var in self.playthrough_vars.items():
            var.set(p_id in playthroughs_in_pool)

        # 还原特殊关
        has_fun = any((lvl - 1) % 10 + 1 == FUN_SUB for lvl, _ in normalized)
        self.var_include_fun.set(has_fun)
        has_conveyor = any((lvl - 1) % 10 + 1 == CONVEYOR_SUB for lvl, _ in normalized)
        self.var_include_conveyor.set(has_conveyor)

        # 还原选卡配合
        if seed_selection_mode in ("adaptive", "adaptive_shuffled"):
            self.var_deck_strategy.set("自动针对僵尸")
            self.var_shuffle_slots.set(seed_selection_mode == "adaptive_shuffled")
        elif seed_selection_mode == "fixed_shuffled":
            self.var_deck_strategy.set("每关预设卡组")
            self.var_shuffle_slots.set(True)
        elif seed_selection_mode == "fixed":
            self.var_deck_strategy.set("每关预设卡组")
            self.var_shuffle_slots.set(False)

        # 检验生成的关卡池是否与传入的完全匹配，若有非规则特殊自定义则保留显式覆盖
        reconstructed = generate_mixed_level_pool(
            scenes=[s for s, v in self.scene_vars.items() if v.get()],
            playthroughs=[p for p, v in self.playthrough_vars.items() if v.get()],
            include_fun=self.var_include_fun.get(),
            include_conveyor=self.var_include_conveyor.get(),
        )
        if reconstructed == normalized:
            self._explicit_level_pool = None
        else:
            self._explicit_level_pool = normalized

        self.mixed_container.pack(fill=tk.X, pady=(0, 6))
        self._handle_change()

    def set_busy(self, busy: bool) -> None:
        """根据训练运行或恢复锁定状态，禁用/启用所有交互控件"""
        for w in getattr(self, "_interactive_widgets", []):
            try:
                if isinstance(w, ttk.Combobox):
                    w.config(state="disabled" if busy else "readonly")
                else:
                    w.config(state="disabled" if busy else "normal")
            except Exception:
                pass
