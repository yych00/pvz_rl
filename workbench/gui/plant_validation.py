# -*- coding: utf-8 -*-
"""GUI controls for Plant legacy/native dual-legality verification."""

from __future__ import annotations

import ctypes
import json
import tkinter as tk
from typing import Any


_STATS_SYMBOL = "pvz_gym_get_plant_legality_compare_json"
_RESET_SYMBOL = "pvz_gym_reset_plant_legality_compare_stats"

_PLANTING_REASON_NAMES = {
    0: "PLANTING_OK",
    1: "PLANTING_NOT_HERE",
    2: "PLANTING_ONLY_ON_GRAVES",
    3: "PLANTING_ONLY_IN_POOL",
    4: "PLANTING_ONLY_ON_GROUND",
    5: "PLANTING_NEEDS_POT",
    6: "PLANTING_NOT_ON_ART",
    7: "PLANTING_NOT_PASSED_LINE",
    8: "PLANTING_NEEDS_UPGRADE",
    9: "PLANTING_NOT_ON_GRAVE",
    10: "PLANTING_NOT_ON_CRATER",
    11: "PLANTING_NOT_ON_WATER",
    12: "PLANTING_NEEDS_GROUND",
    13: "PLANTING_NEEDS_SLEEPING",
}


def install_plant_dual_validation_gui(gui_cls: Any, ui: Any) -> None:
    """Expose accumulated legacy-vs-Board::CanPlantAt comparison evidence."""
    if getattr(gui_cls, "_plant_dual_validation_gui_installed", False):
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

    def _bind_dual_symbols(self):
        dll = getattr(self.client, "dll", None)
        if dll is None:
            raise RuntimeError("DLL 未加载")
        if not hasattr(dll, _STATS_SYMBOL) or not hasattr(dll, _RESET_SYMBOL):
            raise RuntimeError("当前 DLL 未导出 Plant 双轨校验接口，请重新编译 pvz_gym_native.dll")

        stats_fn = getattr(dll, _STATS_SYMBOL)
        stats_fn.argtypes = [ctypes.c_char_p, ctypes.c_int]
        stats_fn.restype = ctypes.c_int

        reset_fn = getattr(dll, _RESET_SYMBOL)
        reset_fn.argtypes = []
        reset_fn.restype = None
        return stats_fn, reset_fn

    def _read_plant_dual_stats(self) -> dict:
        stats_fn, _ = self._bind_plant_dual_symbols()
        buf = ctypes.create_string_buffer(8192)
        written = int(stats_fn(buf, len(buf)))
        if written < 0:
            raise RuntimeError("Plant 双轨统计 JSON 缓冲区不足")
        raw = buf.value.decode("utf-8", errors="replace")
        if not raw:
            raise RuntimeError("Plant 双轨统计为空")
        return json.loads(raw)

    def _format_plant_dual_stats(self, stats: dict) -> str:
        addr = int(stats.get("last_native_address", 0) or 0)
        compared = int(stats.get("compared", 0) or 0)
        if compared == 0:
            status_line = "status            NOT TESTED"
        elif int(stats.get("mismatched", 0) or 0):
            status_line = "status            SPATIAL DIFFERENCE FOUND"
        elif int(stats.get("native_unavailable", 0) or 0):
            status_line = "status            NATIVE UNAVAILABLE"
        else:
            status_line = "status            ALL MATCHED"

        lines = [
            "========== PLANT DUAL VALIDATION ==========",
            status_line,
            f"mode              {stats.get('mode', 'unknown')}",
            f"authority         {stats.get('authoritative', 'unknown')}",
            f"native            {stats.get('native_spatial', 'unknown')}",
            f"native address    0x{addr:08X}" if addr else "native address    not tested yet",
            "",
            f"actions seen      {int(stats.get('actions_seen', 0)):>8}",
            f"compared          {compared:>8}",
            f"matched           {int(stats.get('matched', 0)):>8}",
            f"mismatched        {int(stats.get('mismatched', 0)):>8}",
            f"native unavailable{int(stats.get('native_unavailable', 0)):>8}",
            f"original calls    {int(stats.get('original_calls', 0)):>8}",
            f"mask batches      {int(stats.get('mask_batches', 0)):>8}",
            f"single checks     {int(stats.get('single_checks', 0)):>8}",
            f"consecutive match {int(stats.get('consecutive_matches', 0)):>8}",
        ]

        mismatch = stats.get("last_mismatch")
        if mismatch:
            reason = int(mismatch.get("planting_reason", -1))
            reason_name = _PLANTING_REASON_NAMES.get(reason, "UNKNOWN_PLANTING_REASON")
            lines.extend([
                "",
                "[LAST MISMATCH]",
                (
                    f"slot={mismatch.get('slot')} row={mismatch.get('row')} "
                    f"col={mismatch.get('col')} plant_type={mismatch.get('plant_type')}"
                ),
                (
                    f"legacy={'LEGAL' if mismatch.get('legacy_accepted') else 'ILLEGAL'} "
                    f"error={mismatch.get('legacy_error')}"
                ),
                f"native={'LEGAL' if mismatch.get('native_accepted') else 'ILLEGAL'}",
                f"reason={reason} {reason_name}",
            ])
        else:
            lines.extend(["", "[LAST MISMATCH] none"])

        return "\n".join(lines)

    def cmd_refresh_plant_dual_validation(self):
        try:
            stats = self._read_plant_dual_stats()
            output = self._format_plant_dual_stats(stats)
            self.log("HEADER", "\n" + output)

            mismatch_count = int(stats.get("mismatched", 0) or 0)
            compared = int(stats.get("compared", 0) or 0)
            unavailable = int(stats.get("native_unavailable", 0) or 0)
            runtime = self.client.get_runtime_state()
            mode = int(runtime.get("game_mode", -1))
            level = int(runtime.get("level_id", -1))
            native_current = (mode == 0 and level in (5, 10)) or mode in (17, 33)
            authority_text = "当前关卡按原生空间判断" if native_current else "当前关卡按手写规则判断"
            if hasattr(self, "lbl_plant_dual_status"):
                if compared == 0:
                    self.lbl_plant_dual_status.config(
                        text="状态: 尚未测试 | 先打印一次当前合法动作摘要",
                        fg=ui.COLOR_BLUE,
                    )
                elif mismatch_count:
                    self.lbl_plant_dual_status.config(
                        text=f"状态: 发现差异 {mismatch_count} | compared={compared} | {authority_text}",
                        fg=ui.COLOR_GOLD if native_current else ui.COLOR_RED,
                    )
                elif unavailable:
                    self.lbl_plant_dual_status.config(
                        text=(
                            f"状态: 无差异但 native unavailable={unavailable} | "
                            f"compared={compared}"
                        ),
                        fg=ui.COLOR_GOLD,
                    )
                else:
                    self.lbl_plant_dual_status.config(
                        text=f"状态: 当前全部一致 | compared={compared}",
                        fg=ui.COLOR_GREEN,
                    )
        except Exception as exc:
            self.log("FAIL", f"[PlantDualValidation] {exc}")
            if hasattr(self, "lbl_plant_dual_status"):
                self.lbl_plant_dual_status.config(
                    text=f"状态: 读取失败 | {exc}", fg=ui.COLOR_RED
                )

    def cmd_reset_plant_dual_validation(self):
        try:
            _, reset_fn = self._bind_plant_dual_symbols()
            reset_fn()
            self.log("INFO", "[PlantDualValidation] 累计对比统计已清零")
            if hasattr(self, "lbl_plant_dual_status"):
                self.lbl_plant_dual_status.config(
                    text="状态: 已清零 | 等待新的 Plant 合法性对比",
                    fg=ui.COLOR_BLUE,
                )
        except Exception as exc:
            self.log("FAIL", f"[PlantDualValidation RESET] {exc}")

    def _build_left_panel_with_plant_dual(self, parent):
        original_build_left_panel(self, parent)
        scrollable_frame = _find_scrollable_frame(parent)
        if scrollable_frame is None:
            return

        children = list(scrollable_frame.winfo_children())
        summary_card = _find_card(scrollable_frame, "BattleAction 当前合法动作摘要")
        card = self._create_card(scrollable_frame, "PLANT 双轨合法性校验")
        card.pack_forget()
        if summary_card is not None:
            card.pack(fill=tk.X, pady=(0, 10), after=summary_card)
        elif children:
            card.pack(fill=tk.X, pady=(0, 10), after=children[0])
        else:
            card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            card,
            text=(
                "冒险 1-5、1-10 与坚果保龄球挑战使用原版 Board::CanPlantAt "
                "判断种植位置；其他关卡继续使用手写规则并记录双轨差异。"
            ),
            font=("Microsoft YaHei UI", 8),
            fg=ui.CLR_TEXT_MUTED,
            bg=ui.CLR_PANEL,
            justify="left",
            wraplength=700,
        ).pack(anchor="w", padx=12, pady=(0, 8))

        self.lbl_plant_dual_status = tk.Label(
            card,
            text="状态: 按当前关卡确定权威规则 | 等待对比数据",
            font=ui.FONT_BOLD,
            fg=ui.COLOR_BLUE,
            bg=ui.CLR_PANEL,
            justify="left",
        )
        self.lbl_plant_dual_status.pack(anchor="w", padx=12, pady=(0, 8))

        row = tk.Frame(card, bg=ui.CLR_PANEL)
        row.pack(fill=tk.X, padx=12, pady=(0, 12))

        self.btn_plant_dual_refresh = ui.CyberButton(
            row,
            "[查看累计对比统计]",
            self.cmd_refresh_plant_dual_validation,
            bg_color=ui.COLOR_GREEN,
            fg_color="#ffffff",
            height=34,
        )
        self.btn_plant_dual_refresh.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        self.btn_plant_dual_reset = ui.CyberButton(
            row,
            "[清零重新验证]",
            self.cmd_reset_plant_dual_validation,
            bg_color=ui.COLOR_GOLD,
            fg_color="#ffffff",
            height=34,
        )
        self.btn_plant_dual_reset.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

    gui_cls._bind_plant_dual_symbols = _bind_dual_symbols
    gui_cls._read_plant_dual_stats = _read_plant_dual_stats
    gui_cls._format_plant_dual_stats = _format_plant_dual_stats
    gui_cls.cmd_refresh_plant_dual_validation = cmd_refresh_plant_dual_validation
    gui_cls.cmd_reset_plant_dual_validation = cmd_reset_plant_dual_validation
    gui_cls._build_left_panel = _build_left_panel_with_plant_dual
    gui_cls._plant_dual_validation_gui_installed = True
