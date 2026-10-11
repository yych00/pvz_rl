# -*- coding: utf-8 -*-
"""Compact legal-action summary for the native BattleAction GUI."""

from __future__ import annotations

import threading
import tkinter as tk
from collections import Counter
from typing import Any


ACTION_SPACE_SIZE = 757
PLANT_BASE = 1
MAX_CARD_SLOTS = 10
MAX_ROWS = 6
COLS = 9
SHOVEL_NORMAL_BASE = 541
SHOVEL_PUMPKIN_BASE = 595
COB_BASE = 649
COB_TARGET_CELLS = MAX_ROWS * COLS

PLANT_NAMES = (
    "Peashooter", "Sunflower", "Cherry Bomb", "Wall-nut", "Potato Mine",
    "Snow Pea", "Chomper", "Repeater", "Puff-shroom", "Sun-shroom",
    "Fume-shroom", "Grave Buster", "Hypno-shroom", "Scaredy-shroom",
    "Ice-shroom", "Doom-shroom", "Lily Pad", "Squash", "Threepeater",
    "Tangle Kelp", "Jalapeno", "Spikeweed", "Torchwood", "Tall-nut",
    "Sea-shroom", "Plantern", "Cactus", "Blover", "Split Pea", "Starfruit",
    "Pumpkin", "Magnet-shroom", "Cabbage-pult", "Flower Pot", "Kernel-pult",
    "Coffee Bean", "Garlic", "Umbrella Leaf", "Marigold", "Melon-pult",
    "Gatling Pea", "Twin Sunflower", "Gloom-shroom", "Cattail", "Winter Melon",
    "Gold Magnet", "Spikerock", "Cob Cannon",
    "Imitater", "Explode-o-nut", "Giant Wall-nut",
)

CONVEYOR_ADVENTURE_LEVELS = frozenset((5, 10, 20, 25, 30, 40, 45, 50))
CONVEYOR_CHALLENGE_MODES = frozenset((17, 21, 25, 26, 27, 33, 35, 46, 47, 48))



def _plant_name(plant_type: int) -> str:
    if 0 <= plant_type < len(PLANT_NAMES):
        return PLANT_NAMES[plant_type]
    if plant_type < 0:
        return "Unknown"
    return f"PlantType {plant_type}"


def install_battle_action_summary_gui(gui_cls: Any, ui: Any) -> None:
    """Add a compact, read-only summary of every currently legal battle action."""
    if getattr(gui_cls, "_battle_action_summary_gui_installed", False):
        return

    original_build_left_panel = gui_cls._build_left_panel

    def _find_scrollable_frame(parent):
        canvas = next((w for w in parent.winfo_children() if isinstance(w, tk.Canvas)), None)
        if canvas is None:
            return None
        return next((w for w in canvas.winfo_children() if isinstance(w, tk.Frame)), None)

    def _find_card(frame, title_fragment: str):
        for child in frame.winfo_children():
            for grandchild in child.winfo_children():
                if isinstance(grandchild, tk.Label) and title_fragment in str(grandchild.cget("text")):
                    return child
        return None

    def _build_left_panel_with_action_summary(self, parent):
        original_build_left_panel(self, parent)
        scrollable_frame = _find_scrollable_frame(parent)
        if scrollable_frame is None:
            return

        children = list(scrollable_frame.winfo_children())
        manual_card = _find_card(scrollable_frame, "BattleAction 手动实机验证")
        card = self._create_card(scrollable_frame, "BattleAction 当前合法动作摘要")
        card.pack_forget()
        if manual_card is not None:
            card.pack(fill=tk.X, pady=(0, 10), after=manual_card)
        elif children:
            card.pack(fill=tk.X, pady=(0, 10), after=children[0])
        else:
            card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            card,
            text="按 Native Action Mask 汇总，不展开坐标。PLANT 同时列出卡槽与植物类型；传送带卡按卡包可用性判断。SHOVEL 按可铲植物类型累计。",
            font=("Microsoft YaHei UI", 8),
            fg=ui.CLR_TEXT_MUTED,
            bg=ui.CLR_PANEL,
            justify="left",
            wraplength=700,
        ).pack(anchor="w", padx=12, pady=(0, 8))

        self.lbl_legal_action_summary_status = tk.Label(
            card,
            text="状态: 就绪 | 战斗 UI=3 时可读取",
            font=ui.FONT_BOLD,
            fg=ui.COLOR_BLUE,
            bg=ui.CLR_PANEL,
            justify="left",
        )
        self.lbl_legal_action_summary_status.pack(anchor="w", padx=12, pady=(0, 8))

        self.btn_legal_action_summary = ui.CyberButton(
            card,
            "[打印当前所有合法动作摘要]",
            self.cmd_print_legal_action_summary,
            bg_color=ui.COLOR_GREEN,
            fg_color="#ffffff",
            height=34,
        )
        self.btn_legal_action_summary.pack(fill=tk.X, padx=12, pady=(0, 12))

    def _collect_legal_action_summary(self):
        ui_code, ui_desc = self.client.get_ui()
        if ui_code != 3:
            raise RuntimeError(f"当前不是战斗 UI: {ui_desc}")

        code, mask, count, msg = self.client.get_action_mask(ACTION_SPACE_SIZE)
        if code != 0 or count != ACTION_SPACE_SIZE or len(mask) != ACTION_SPACE_SIZE:
            raise RuntimeError(
                f"读取 757 维 action mask 失败: code={code}, count={count}, len={len(mask)}, {msg}"
            )

        _, obs_msg, obs = self.client.get_observation_json()
        if not obs:
            raise RuntimeError(f"读取 Observation 失败: {obs_msg}")

        cards_by_slot = {
            int(card.get("slot", -1)): card
            for card in obs.get("cards", [])
            if 0 <= int(card.get("slot", -1)) < MAX_CARD_SLOTS
        }
        global_info = obs.get("global", {})
        card_system = str(global_info.get("card_system", "") or "")
        mode_name = str(global_info.get("game_mode_name", "") or "")
        if "is_conveyor" in global_info:
            conveyor = bool(global_info.get("is_conveyor"))
        elif card_system:
            conveyor = card_system == "conveyor"
        else:
            runtime = self.client.get_runtime_state()
            mode = int(runtime.get("game_mode", -1))
            level = int(global_info.get("level_id", -1))
            conveyor = (
                (mode == 0 and level in CONVEYOR_ADVENTURE_LEVELS)
                or mode in CONVEYOR_CHALLENGE_MODES
            )
        if not card_system:
            card_system = "conveyor" if conveyor else "seed_bank"


        # PLANT: same plant type can exist in more than one slot; aggregate all
        # legal (slot,row,col) actions under the plant type.
        plant_counts: Counter[int] = Counter()
        slot_counts: Counter[int] = Counter()
        for slot in range(MAX_CARD_SLOTS):
            plant_type = int(cards_by_slot.get(slot, {}).get("plant_type", -1))
            for row in range(MAX_ROWS):
                for col in range(COLS):
                    idx = PLANT_BASE + ((slot * MAX_ROWS + row) * COLS + col)
                    if mask[idx]:
                        plant_counts[plant_type] += 1
                        slot_counts[slot] += 1
        plant_total = sum(plant_counts.values())
        card_slots = [
            {
                "slot": slot,
                "plant_type": int(card.get("plant_type", -1)),
                "ready": bool(card.get("ready", False)),
                "legal": slot_counts[slot],
                "cost": int(card.get("cost", 0)),
                "cooldown_remaining": int(card.get("cooldown_remaining", 0)),
            }
            for slot, card in sorted(cards_by_slot.items())
        ]

        # Index observed plants by cell/layer so a legal shovel action can be
        # reported as the plant that would actually be removed.
        cells: dict[tuple[int, int], dict[int, list[dict]]] = {}
        for plant in obs.get("plants", []):
            row = int(plant.get("row", -1))
            col = int(plant.get("col", -1))
            layer = int(plant.get("layer", -1))
            if row < 0 or col < 0:
                continue
            cells.setdefault((row, col), {}).setdefault(layer, []).append(plant)

        shovel_counts: Counter[int] = Counter()
        unknown_shovel_targets = 0
        for row in range(MAX_ROWS):
            for col in range(COLS):
                layers = cells.get((row, col), {})

                normal_idx = SHOVEL_NORMAL_BASE + row * COLS + col
                if mask[normal_idx]:
                    targets = layers.get(1) or layers.get(0) or []
                    if targets:
                        shovel_counts[int(targets[0].get("type", -1))] += 1
                    else:
                        unknown_shovel_targets += 1

                pumpkin_idx = SHOVEL_PUMPKIN_BASE + row * COLS + col
                if mask[pumpkin_idx]:
                    targets = layers.get(2) or []
                    if targets:
                        shovel_counts[int(targets[0].get("type", -1))] += 1
                    else:
                        unknown_shovel_targets += 1

        shovel_total = sum(shovel_counts.values()) + unknown_shovel_targets

        # COB_FIRE has one action per target cell; the cannon is selected natively.
        rows = max(1, min(MAX_ROWS, int(obs.get("global", {}).get("grid_rows", 5) or 5)))
        legal_targets = set()
        cob_total = 0
        for target_row in range(rows):
            for target_col in range(COLS):
                idx = COB_BASE + target_row * COLS + target_col
                if mask[idx]:
                    cob_total += 1
                    legal_targets.add((target_row, target_col))

        return {
            "plant_counts": plant_counts,
            "plant_total": plant_total,
            "card_slots": card_slots,
            "conveyor": conveyor,
            "card_system": card_system,
            "game_mode_name": mode_name,
            "shovel_counts": shovel_counts,
            "unknown_shovel_targets": unknown_shovel_targets,
            "shovel_total": shovel_total,
            "ready_cannons": "at least one" if cob_total else "none",
            "target_cells": len(legal_targets),
            "cob_total": cob_total,
        }

    def _format_legal_action_summary(self, summary: dict) -> str:
        mode_header = f"Mode: {summary['game_mode_name']}" if summary.get("game_mode_name") else ""
        card_mode_text = (
            summary.get("card_system", "").upper()
            if summary.get("card_system")
            else ("CONVEYOR" if summary["conveyor"] else "SEED BANK / DROPS")
        )
        lines = [
            "========== LEGAL ACTION SUMMARY ==========",
        ]
        if mode_header:
            lines.append(mode_header)
        lines.extend([
            f"Card system: {card_mode_text}",
            "Counts come from the native action mask.",
            "",
            "[CARD SLOTS]",
        ])
        for card in summary["card_slots"]:
            detail = "packet" if summary["conveyor"] else (
                f"sun={card['cost']} cd={card['cooldown_remaining']}"
            )
            lines.append(
                f"slot {card['slot']:>2}  {_plant_name(card['plant_type']):<20} "
                f"ready={int(card['ready'])} legal={card['legal']:>3}  {detail}"
            )
        lines.extend(["", "[PLANT BY TYPE]"])
        plant_counts = summary["plant_counts"]
        for plant_type in sorted(plant_counts, key=lambda t: (_plant_name(t), t)):
            lines.append(f"{_plant_name(plant_type):<20} {plant_counts[plant_type]:>5}")
        lines.append(f"{'TOTAL':<20} {summary['plant_total']:>5}")

        lines.extend(["", "[SHOVEL]"])
        shovel_counts = summary["shovel_counts"]
        for plant_type in sorted(shovel_counts, key=lambda t: (_plant_name(t), t)):
            lines.append(f"{_plant_name(plant_type):<20} {shovel_counts[plant_type]:>5}")
        if summary["unknown_shovel_targets"]:
            lines.append(f"{'Unknown target':<20} {summary['unknown_shovel_targets']:>5}")
        lines.append(f"{'TOTAL':<20} {summary['shovel_total']:>5}")

        lines.extend([
            "",
            "[COB_FIRE]",
            f"{'Ready cannon':<20} {summary['ready_cannons']:>5}",
            f"{'Target cells':<20} {summary['target_cells']:>5}",
            f"{'TOTAL':<20} {summary['cob_total']:>5}",
            "",
            "------------------------------------------",
            f"{'PLANT':<20} {summary['plant_total']:>5}",
            f"{'SHOVEL':<20} {summary['shovel_total']:>5}",
            f"{'COB_FIRE':<20} {summary['cob_total']:>5}",
            f"{'TOTAL':<20} {summary['plant_total'] + summary['shovel_total'] + summary['cob_total']:>5}",
        ])
        return "\n".join(lines)

    def cmd_print_legal_action_summary(self):
        if getattr(self, "_legal_action_summary_running", False):
            self.log("WARN", "[LegalActionSummary] 正在统计，请勿重复触发")
            return

        self._legal_action_summary_running = True
        if hasattr(self, "btn_legal_action_summary"):
            self.btn_legal_action_summary.set_disabled(True)
        if hasattr(self, "lbl_legal_action_summary_status"):
            self.lbl_legal_action_summary_status.config(text="状态: 统计中...", fg=ui.COLOR_GOLD)

        def _worker():
            was_paused = False
            paused_by_summary = False
            try:
                ui_code, ui_desc = self.client.get_ui()
                if ui_code != 3:
                    raise RuntimeError(f"当前不是战斗 UI: {ui_desc}")

                was_paused = bool(self.client.is_paused())
                if not was_paused:
                    paused_by_summary = bool(self.client.pause())

                summary = self._collect_legal_action_summary()
                output = self._format_legal_action_summary(summary)

                def _success():
                    self.log("HEADER", "\n" + output)
                    if hasattr(self, "lbl_legal_action_summary_status"):
                        total = summary["plant_total"] + summary["shovel_total"] + summary["cob_total"]
                        self.lbl_legal_action_summary_status.config(
                            text=f"状态: 完成 | 当前合法战略动作 {total}",
                            fg=ui.COLOR_GREEN,
                        )

                self.root.after(0, _success)
            except Exception as exc:
                def _failure(err=exc):
                    self.log("FAIL", f"[LegalActionSummary] {err}")
                    if hasattr(self, "lbl_legal_action_summary_status"):
                        self.lbl_legal_action_summary_status.config(
                            text=f"状态: 读取失败 | {err}",
                            fg=ui.COLOR_RED,
                        )
                self.root.after(0, _failure)
            finally:
                if paused_by_summary and not was_paused:
                    self.client.resume()

                def _finish():
                    self._legal_action_summary_running = False
                    if hasattr(self, "btn_legal_action_summary"):
                        self.btn_legal_action_summary.set_disabled(False)
                    self.refresh_status()

                self.root.after(0, _finish)

        threading.Thread(target=_worker, daemon=True).start()

    gui_cls._build_left_panel = _build_left_panel_with_action_summary
    gui_cls._collect_legal_action_summary = _collect_legal_action_summary
    gui_cls._format_legal_action_summary = _format_legal_action_summary
    gui_cls.cmd_print_legal_action_summary = cmd_print_legal_action_summary
    gui_cls._battle_action_summary_gui_installed = True
