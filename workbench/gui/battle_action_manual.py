# -*- coding: utf-8 -*-
"""Manual, visually verifiable BattleAction controls for the native test GUI."""

from __future__ import annotations

import tkinter as tk
from typing import Any


ACTION_NOOP = 0
ACTION_PLANT = 1
ACTION_SHOVEL = 2
ACTION_COB_FIRE = 3
SHOVEL_NORMAL = 0
SHOVEL_PUMPKIN = 1


def install_battle_action_manual_gui(gui_cls: Any, ui: Any) -> None:
    """Expose explicit one-shot native actions so the user can watch PvZ and verify them."""
    if getattr(gui_cls, "_battle_action_manual_gui_installed", False):
        return

    original_build_left_panel = gui_cls._build_left_panel

    def _find_scrollable_frame(parent):
        canvas = next((w for w in parent.winfo_children() if isinstance(w, tk.Canvas)), None)
        if canvas is None:
            return None
        return next((w for w in canvas.winfo_children() if isinstance(w, tk.Frame)), None)

    def _combo(parent, values, default, width=5):
        box = ui.ttk.Combobox(parent, values=[str(v) for v in values], width=width, state="readonly")
        box.set(str(default))
        return box

    def _label(parent, text):
        return tk.Label(parent, text=text, font=ui.FONT_BOLD, fg=ui.CLR_TEXT_MAIN, bg=ui.CLR_PANEL)

    def _make_buttons(self, parent, action_name, row_num):
        ui.CyberButton(
            parent,
            "检查合法性",
            lambda name=action_name: self.cmd_manual_action_check(name),
            bg_color=ui.CLR_CARD,
            fg_color=ui.COLOR_BLUE,
            height=30,
            padding_x=10,
        ).grid(row=row_num, column=0, padx=4, pady=4, sticky="ew")
        ui.CyberButton(
            parent,
            "真实执行一次",
            lambda name=action_name: self.cmd_manual_action_execute(name),
            bg_color=ui.COLOR_GOLD,
            fg_color=ui.CLR_TEXT_DARK,
            height=30,
            padding_x=10,
        ).grid(row=row_num, column=1, padx=4, pady=4, sticky="ew")

    def _build_left_panel_with_manual_actions(self, parent):
        original_build_left_panel(self, parent)
        scrollable_frame = _find_scrollable_frame(parent)
        if scrollable_frame is None:
            return

        # Insert immediately after the automatic BattleAction test card when present.
        children = list(scrollable_frame.winfo_children())
        battle_card = next(
            (
                child for child in children
                if any(
                    isinstance(grandchild, tk.Label)
                    and "BattleAction 战斗动作单元测试" in str(grandchild.cget("text"))
                    for grandchild in child.winfo_children()
                )
            ),
            None,
        )

        card = self._create_card(scrollable_frame, "BattleAction 实机验证")
        card.pack_forget()
        if battle_card is not None:
            card.pack(fill=tk.X, pady=(0, 10), after=battle_card)
        elif children:
            card.pack(fill=tk.X, pady=(0, 10), after=children[0])
        else:
            card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            card,
            text=(
                "这里不做自动 PASS 代替肉眼判断。先点“检查合法性”，再点“真实执行一次”；"
                "动作只执行一次，不自动连发、不自动暂停。请直接观察 PvZ 窗口验证结果。"
            ),
            font=("Microsoft YaHei UI", 8),
            fg=ui.CLR_TEXT_MUTED,
            bg=ui.CLR_PANEL,
            justify="left",
            wraplength=560,
        ).pack(anchor="w", padx=12, pady=(0, 6))

        self.lbl_manual_action_status = tk.Label(
            card,
            text="状态: 就绪 | 要求游戏处于战斗 UI=3",
            font=ui.FONT_BOLD,
            fg=ui.COLOR_BLUE,
            bg=ui.CLR_PANEL,
            justify="left",
        )
        self.lbl_manual_action_status.pack(anchor="w", padx=12, pady=(0, 8))

        # NOOP -----------------------------------------------------------------
        noop = tk.Frame(card, bg=ui.CLR_PANEL)
        noop.pack(fill=tk.X, padx=12, pady=(2, 6))
        tk.Label(noop, text="NOOP", width=12, anchor="w", font=ui.FONT_TITLE, fg=ui.CLR_TEXT_MAIN, bg=ui.CLR_PANEL).pack(side=tk.LEFT)
        tk.Label(noop, text="无参数；用于确认执行链本身", font=("Microsoft YaHei UI", 8), fg=ui.CLR_TEXT_MUTED, bg=ui.CLR_PANEL).pack(side=tk.LEFT, padx=(4, 10))
        ui.CyberButton(noop, "检查", lambda: self.cmd_manual_action_check("noop"), bg_color=ui.CLR_CARD, fg_color=ui.COLOR_BLUE, height=28, padding_x=10).pack(side=tk.RIGHT, padx=3)
        ui.CyberButton(noop, "执行一次", lambda: self.cmd_manual_action_execute("noop"), bg_color=ui.COLOR_GOLD, fg_color=ui.CLR_TEXT_DARK, height=28, padding_x=10).pack(side=tk.RIGHT, padx=3)

        # PLANT ----------------------------------------------------------------
        plant_box = tk.Frame(card, bg=ui.CLR_PANEL, highlightbackground=ui.CLR_BORDER, highlightthickness=1)
        plant_box.pack(fill=tk.X, padx=12, pady=6)
        tk.Label(plant_box, text="PLANT(slot, row, col)", font=ui.FONT_TITLE, fg=ui.CLR_TEXT_MAIN, bg=ui.CLR_PANEL).pack(anchor="w", padx=8, pady=(7, 4))
        plant_args = tk.Frame(plant_box, bg=ui.CLR_PANEL)
        plant_args.pack(fill=tk.X, padx=8, pady=(0, 4))
        _label(plant_args, "slot").pack(side=tk.LEFT)
        self.cmb_manual_plant_slot = _combo(plant_args, range(10), 0, 4)
        self.cmb_manual_plant_slot.pack(side=tk.LEFT, padx=(4, 10))
        _label(plant_args, "row").pack(side=tk.LEFT)
        self.cmb_manual_plant_row = _combo(plant_args, range(6), 2, 4)
        self.cmb_manual_plant_row.pack(side=tk.LEFT, padx=(4, 10))
        _label(plant_args, "col").pack(side=tk.LEFT)
        self.cmb_manual_plant_col = _combo(plant_args, range(9), 4, 4)
        self.cmb_manual_plant_col.pack(side=tk.LEFT, padx=(4, 10))
        plant_btns = tk.Frame(plant_box, bg=ui.CLR_PANEL)
        plant_btns.pack(fill=tk.X, padx=8, pady=(0, 8))
        plant_btns.columnconfigure(0, weight=1)
        plant_btns.columnconfigure(1, weight=1)
        _make_buttons(self, plant_btns, "plant", 0)

        # SHOVEL ---------------------------------------------------------------
        shovel_box = tk.Frame(card, bg=ui.CLR_PANEL, highlightbackground=ui.CLR_BORDER, highlightthickness=1)
        shovel_box.pack(fill=tk.X, padx=12, pady=6)
        tk.Label(shovel_box, text="SHOVEL(row, col, target)", font=ui.FONT_TITLE, fg=ui.CLR_TEXT_MAIN, bg=ui.CLR_PANEL).pack(anchor="w", padx=8, pady=(7, 4))
        shovel_args = tk.Frame(shovel_box, bg=ui.CLR_PANEL)
        shovel_args.pack(fill=tk.X, padx=8, pady=(0, 4))
        _label(shovel_args, "row").pack(side=tk.LEFT)
        self.cmb_manual_shovel_row = _combo(shovel_args, range(6), 2, 4)
        self.cmb_manual_shovel_row.pack(side=tk.LEFT, padx=(4, 10))
        _label(shovel_args, "col").pack(side=tk.LEFT)
        self.cmb_manual_shovel_col = _combo(shovel_args, range(9), 4, 4)
        self.cmb_manual_shovel_col.pack(side=tk.LEFT, padx=(4, 10))
        _label(shovel_args, "target").pack(side=tk.LEFT)
        self.cmb_manual_shovel_target = ui.ttk.Combobox(shovel_args, values=["NORMAL", "PUMPKIN"], width=10, state="readonly")
        self.cmb_manual_shovel_target.set("NORMAL")
        self.cmb_manual_shovel_target.pack(side=tk.LEFT, padx=(4, 10))
        shovel_btns = tk.Frame(shovel_box, bg=ui.CLR_PANEL)
        shovel_btns.pack(fill=tk.X, padx=8, pady=(0, 8))
        shovel_btns.columnconfigure(0, weight=1)
        shovel_btns.columnconfigure(1, weight=1)
        _make_buttons(self, shovel_btns, "shovel", 0)

        # COB ------------------------------------------------------------------
        cob_box = tk.Frame(card, bg=ui.CLR_PANEL, highlightbackground=ui.CLR_BORDER, highlightthickness=1)
        cob_box.pack(fill=tk.X, padx=12, pady=6)
        tk.Label(cob_box, text="COB_FIRE(source_row, source_col, target_row, target_col)", font=ui.FONT_TITLE, fg=ui.CLR_TEXT_MAIN, bg=ui.CLR_PANEL).pack(anchor="w", padx=8, pady=(7, 4))
        cob_args1 = tk.Frame(cob_box, bg=ui.CLR_PANEL)
        cob_args1.pack(fill=tk.X, padx=8, pady=(0, 3))
        _label(cob_args1, "source row").pack(side=tk.LEFT)
        self.cmb_manual_cob_source_row = _combo(cob_args1, range(6), 2, 4)
        self.cmb_manual_cob_source_row.pack(side=tk.LEFT, padx=(4, 10))
        _label(cob_args1, "source col").pack(side=tk.LEFT)
        self.cmb_manual_cob_source_col = _combo(cob_args1, range(8), 0, 4)
        self.cmb_manual_cob_source_col.pack(side=tk.LEFT, padx=(4, 10))
        cob_args2 = tk.Frame(cob_box, bg=ui.CLR_PANEL)
        cob_args2.pack(fill=tk.X, padx=8, pady=(0, 4))
        _label(cob_args2, "target row").pack(side=tk.LEFT)
        self.cmb_manual_cob_target_row = _combo(cob_args2, range(6), 2, 4)
        self.cmb_manual_cob_target_row.pack(side=tk.LEFT, padx=(4, 10))
        _label(cob_args2, "target col").pack(side=tk.LEFT)
        self.cmb_manual_cob_target_col = _combo(cob_args2, range(9), 8, 4)
        self.cmb_manual_cob_target_col.pack(side=tk.LEFT, padx=(4, 10))
        cob_btns = tk.Frame(cob_box, bg=ui.CLR_PANEL)
        cob_btns.pack(fill=tk.X, padx=8, pady=(0, 8))
        cob_btns.columnconfigure(0, weight=1)
        cob_btns.columnconfigure(1, weight=1)
        _make_buttons(self, cob_btns, "cob", 0)

        tk.Label(
            card,
            text="提示：如果游戏当前是暂停状态，需要 Resume 后观察。",
            font=("Microsoft YaHei UI", 8),
            fg=ui.COLOR_GOLD,
            bg=ui.CLR_PANEL,
            justify="left",
            wraplength=560,
        ).pack(anchor="w", padx=12, pady=(2, 10))

    def _manual_action_spec(self, name: str):
        if name == "noop":
            return ACTION_NOOP, {}, "NOOP"
        if name == "plant":
            slot = int(self.cmb_manual_plant_slot.get())
            row = int(self.cmb_manual_plant_row.get())
            col = int(self.cmb_manual_plant_col.get())
            return ACTION_PLANT, {"card_slot": slot, "row": row, "col": col}, f"PLANT(slot={slot}, row={row}, col={col})"
        if name == "shovel":
            row = int(self.cmb_manual_shovel_row.get())
            col = int(self.cmb_manual_shovel_col.get())
            target_name = self.cmb_manual_shovel_target.get().strip().upper()
            target = SHOVEL_PUMPKIN if target_name == "PUMPKIN" else SHOVEL_NORMAL
            return ACTION_SHOVEL, {"row": row, "col": col, "shovel_target": target}, f"SHOVEL(row={row}, col={col}, target={target_name})"
        if name == "cob":
            source_row = int(self.cmb_manual_cob_source_row.get())
            source_col = int(self.cmb_manual_cob_source_col.get())
            target_row = int(self.cmb_manual_cob_target_row.get())
            target_col = int(self.cmb_manual_cob_target_col.get())
            return ACTION_COB_FIRE, {
                "row": source_row,
                "col": source_col,
                "target_row": target_row,
                "target_col": target_col,
            }, f"COB_FIRE(source=({source_row},{source_col}), target=({target_row},{target_col}))"
        raise ValueError(f"unknown manual action: {name}")

    def _manual_obs_summary(self, action_type: int, kwargs: dict):
        _, _, obs = self.client.get_observation_json()
        if not obs:
            return "Observation unavailable"
        global_state = obs.get("global", {})
        parts = [
            f"clock={global_state.get('game_clock')}",
            f"sun={global_state.get('sun')}",
            f"plants={len(obs.get('plants', []))}",
        ]
        if action_type in (ACTION_PLANT, ACTION_SHOVEL):
            row = kwargs.get("row", -1)
            col = kwargs.get("col", -1)
            cell = [
                (p.get("id"), p.get("type"), p.get("layer"), p.get("hp"))
                for p in obs.get("plants", [])
                if p.get("row") == row and p.get("col") == col
            ]
            parts.append(f"cell({row},{col})={cell}")
        elif action_type == ACTION_COB_FIRE:
            row = kwargs.get("row", -1)
            col = kwargs.get("col", -1)
            cannon = [
                (p.get("id"), p.get("type"), p.get("state"))
                for p in obs.get("plants", [])
                if p.get("row") == row and p.get("col") == col and p.get("type") == 47
            ]
            parts.append(f"cannon({row},{col})={cannon}")
        return " | ".join(parts)

    def cmd_manual_action_check(self, name: str):
        try:
            action_type, kwargs, desc = self._manual_action_spec(name)
        except Exception as exc:
            self.log("FAIL", f"[ManualAction] 参数解析失败: {exc}")
            return

        code, msg = self.client.can_execute_action(action_type, **kwargs)
        legal = code == 0
        tag = "PASS" if legal else "FAIL"
        self.log(tag, f"[ManualAction CHECK] {desc} -> code={code}, message={msg}")
        if hasattr(self, "lbl_manual_action_status"):
            self.lbl_manual_action_status.config(
                text=f"状态: {'合法，可执行' if legal else '当前不可执行'} | {desc} | code={code}",
                fg=ui.COLOR_GREEN if legal else ui.COLOR_RED,
            )

    def cmd_manual_action_execute(self, name: str):
        try:
            action_type, kwargs, desc = self._manual_action_spec(name)
        except Exception as exc:
            self.log("FAIL", f"[ManualAction] 参数解析失败: {exc}")
            return

        ui_code, ui_desc = self.client.get_ui()
        if ui_code != 3:
            self.log("FAIL", f"[ManualAction EXEC] {desc} 已阻止：当前不是战斗 UI=3，而是 {ui_desc}")
            return

        code, msg = self.client.can_execute_action(action_type, **kwargs)
        self.log("PASS" if code == 0 else "FAIL", f"[ManualAction PRECHECK] {desc} -> code={code}, message={msg}")
        if code != 0:
            if hasattr(self, "lbl_manual_action_status"):
                self.lbl_manual_action_status.config(text=f"状态: 未执行 | precheck code={code} | {desc}", fg=ui.COLOR_RED)
            return

        before = self._manual_obs_summary(action_type, kwargs)
        self.log("MUTED", f"  BEFORE: {before}")

        exec_code, exec_msg = self.client.execute_action(action_type, **kwargs)
        after = self._manual_obs_summary(action_type, kwargs)
        self.log("PASS" if exec_code == 0 else "FAIL", f"[ManualAction EXEC] {desc} -> code={exec_code}, message={exec_msg}")
        self.log("MUTED", f"  AFTER : {after}")

        paused = self.client.is_paused()
        if paused and action_type == ACTION_COB_FIRE and exec_code == 0:
            self.log("WARN", "  当前战斗处于暂停：Cob 已提交，但请点 Resume 后观察炮弹飞行和落点。")

        if hasattr(self, "lbl_manual_action_status"):
            self.lbl_manual_action_status.config(
                text=f"状态: {'已真实执行一次' if exec_code == 0 else '执行失败'} | {desc} | code={exec_code}",
                fg=ui.COLOR_GREEN if exec_code == 0 else ui.COLOR_RED,
            )
        self.refresh_status()

    gui_cls._build_left_panel = _build_left_panel_with_manual_actions
    gui_cls._manual_action_spec = _manual_action_spec
    gui_cls._manual_obs_summary = _manual_obs_summary
    gui_cls.cmd_manual_action_check = cmd_manual_action_check
    gui_cls.cmd_manual_action_execute = cmd_manual_action_execute
    gui_cls._battle_action_manual_gui_installed = True
