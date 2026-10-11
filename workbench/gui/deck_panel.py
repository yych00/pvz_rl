"""Shared, editable deck preview for LIVE tests and Gym training."""

import re
import threading
import tkinter as tk
from tkinter import ttk

from pvz_gym.default_decks import PLANT_NAMES, default_deck_priority, select_default_deck
from pvz_gym.adventure_curriculum import deck_profile, ZOMBIE_NAMES

MODE_LABELS = {"adaptive": "自动针对僵尸", "adaptive_shuffled": "自动针对僵尸（顺序打乱）", "fixed": "每关预设卡组",
               "fixed_shuffled": "每关预设卡组（顺序打乱）",
               "custom": "自定义优先级（旧配置）", "native": "原生指定卡（旧配置）"}
VISIBLE_MODES = ("adaptive", "adaptive_shuffled", "fixed", "fixed_shuffled")


def parse_cards(text):
    values = []
    for token in re.split(r"[,，、;；\s]+", text.strip()):
        if not token:
            continue
        plant = PLANT_NAMES.index(token) if token in PLANT_NAMES else int(token)
        if not 0 <= plant < 40:
            raise ValueError("请选择 0～39 的普通植物；升级卡和模仿者需要另配依赖")
        if plant in values:
            raise ValueError(f"卡牌重复：{PLANT_NAMES[plant]}")
        values.append(plant)
    return tuple(values) or None


class DeckPanel(tk.Frame):
    def __init__(self, parent, ui, context, on_test, *, cards_var=None, legacy=False):
        super().__init__(parent, bg=ui.CLR_CARD, highlightbackground=ui.CLR_BORDER, highlightthickness=1)
        self.ui, self.context, self.on_test = ui, context, on_test
        self.cards_var = cards_var if cards_var is not None else tk.StringVar(value="")
        self.mode_var = tk.StringVar(value=MODE_LABELS["fixed"])
        self.busy = False
        self._preview_initialized = False
        self.profile_key = None
        self.profiles = {}
        self._syncing = False
        top = tk.Frame(self, bg=ui.CLR_CARD)
        top.pack(fill=tk.X, padx=12, pady=(10, 6))
        tk.Label(top, text="卡组配置", bg=ui.CLR_CARD, fg=ui.CLR_TEXT_MAIN,
                 font=ui.FONT_TITLE).pack(side=tk.LEFT)
        self.badge = tk.Label(top, text="", bg="#1c3044", fg=ui.COLOR_BLUE,
                              font=ui.FONT_MAIN, padx=8, pady=3)
        self.badge.pack(side=tk.RIGHT)
        toolbar = tk.Frame(self, bg=ui.CLR_CARD)
        toolbar.pack(fill=tk.X, padx=12, pady=(0, 8))
        self.mode = ttk.Combobox(toolbar, textvariable=self.mode_var,
                                  values=tuple(MODE_LABELS[key] for key in VISIBLE_MODES), state="readonly", width=19)
        self.mode.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.edit_button = ui.CyberButton(toolbar, "编辑卡组", self.edit, height=30,
                                          bg_color="#1f2937", fg_color=ui.COLOR_BLUE)
        self.edit_button.pack(side=tk.LEFT, padx=3)
        self.reset_button = ui.CyberButton(toolbar, "恢复默认", self.reset, height=30,
                                           bg_color=ui.CLR_PANEL, fg_color=ui.CLR_TEXT_MUTED)
        self.reset_button.pack(side=tk.LEFT, padx=3)
        self.test_button = ui.CyberButton(toolbar, "验证选卡", lambda: on_test(self), height=30,
                                          bg_color=ui.COLOR_BLUE, fg_color="#ffffff")
        self.test_button.pack(side=tk.LEFT, padx=(6, 0))
        self.preview_row = tk.Frame(self, bg=ui.CLR_CARD)
        tk.Label(self.preview_row, text="查看预设", bg=ui.CLR_CARD,
                 fg=ui.CLR_TEXT_MUTED, font=ui.FONT_MAIN).pack(side=tk.LEFT, padx=(0, 8))
        self.preview_level_var = tk.StringVar(value="1-1")
        self.preview_playthrough_var = tk.StringVar(value="1")
        self.preview_level = ttk.Combobox(
            self.preview_row, textvariable=self.preview_level_var, state="readonly", width=7,
            values=tuple(f"{(level-1)//10+1}-{(level-1)%10+1}" for level in range(1, 51)))
        self.preview_level.pack(side=tk.LEFT, padx=(0, 8))
        self.preview_playthrough = ttk.Combobox(
            self.preview_row, textvariable=self.preview_playthrough_var,
            state="readonly", width=3, values=("1", "2"))
        self.preview_playthrough.pack(side=tk.LEFT)
        tk.Label(self.preview_row, text="周目 · 仅预览", bg=ui.CLR_CARD,
                 fg=ui.CLR_TEXT_MUTED, font=ui.FONT_MAIN).pack(side=tk.LEFT, padx=8)
        self.chips = tk.Frame(self, bg=ui.CLR_CARD)
        self.chips.pack(fill=tk.X, padx=12)
        for col in range(4):
            self.chips.columnconfigure(col, weight=1, uniform="deck")
        entry_row = self.entry_row = tk.Frame(self, bg=ui.CLR_CARD)
        entry_row.pack(fill=tk.X, padx=12, pady=(8, 0))
        tk.Label(entry_row, text="优先顺序", bg=ui.CLR_CARD, fg=ui.CLR_TEXT_MUTED,
                 font=ui.FONT_MAIN).pack(side=tk.LEFT, padx=(0, 8))
        self.entry = tk.Entry(entry_row, textvariable=self.cards_var, bg="#111a25",
                              fg=ui.CLR_TEXT_MAIN, disabledbackground=ui.CLR_PANEL,
                              disabledforeground=ui.CLR_TEXT_MUTED, insertbackground=ui.COLOR_BLUE,
                              relief=tk.FLAT, highlightthickness=1,
                              highlightbackground=ui.CLR_BORDER, highlightcolor=ui.COLOR_BLUE,
                              font=ui.FONT_MAIN)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.note = tk.Label(self, bg=ui.CLR_CARD, fg=ui.CLR_TEXT_MUTED, font=ui.FONT_MAIN,
                             anchor="w", justify=tk.LEFT)
        self.note.pack(fill=tk.X, padx=12, pady=(6, 10))
        self.note.bind("<Configure>", lambda e: self.note.config(wraplength=max(1, e.width)))
        self.mode_var.trace_add("write", lambda *_: self.refresh())
        self.cards_var.trace_add("write", lambda *_: self.refresh())
        self.preview_level_var.trace_add("write", lambda *_: self.refresh())
        self.preview_playthrough_var.trace_add("write", lambda *_: self.refresh())
        self.refresh()

    def settings(self):
        mode = next(key for key, label in MODE_LABELS.items() if label == self.mode_var.get())
        cards = None if mode in ("adaptive", "adaptive_shuffled", "fixed", "fixed_shuffled") else parse_cards(self.cards_var.get())
        if mode == "custom" and not cards:
            raise ValueError("请先编辑自定义卡组")
        return mode, cards

    def refresh(self):
        if self._syncing:
            return
        level, playthrough = self.context()
        key = (level, playthrough)
        if key != self.profile_key:
            if self.profile_key is not None:
                self.profiles[self.profile_key] = (self.mode_var.get(), self.cards_var.get())
            self.profile_key = key
            mode, cards = self.profiles.get(key, (MODE_LABELS["fixed"], ""))
            self._syncing = True
            try:
                self.mode_var.set(mode)
                self.cards_var.set(cards)
            finally:
                self._syncing = False
        self.profiles[key] = (self.mode_var.get(), self.cards_var.get())
        if self.busy:
            stage, sublevel = map(int, self.preview_level_var.get().split("-"))
            level, playthrough = (stage - 1) * 10 + sublevel, int(self.preview_playthrough_var.get())
        self.badge.config(text=f"{(level-1)//10+1}-{(level-1)%10+1} · {playthrough}周目")
        try:
            mode, cards = ("fixed", None) if self.busy else self.settings()
            shown = cards or default_deck_priority(level, playthrough)
            profile = deck_profile(level, playthrough)
            is_special = bool(profile.special)
            if is_special:
                self.badge.config(text=f"{(level-1)//10+1}-{(level-1)%10+1} · {profile.special}")
                note = f"本关机制：{profile.special}\n{profile.lesson}\n（本关为特殊关卡，由游戏传送带系统自动供给或专属规则派发，无需选卡）"
            elif mode in ("fixed", "fixed_shuffled"):
                new = "、".join(PLANT_NAMES[p] for p in profile.new_plants) or "无新植物"
                threats = "、".join(ZOMBIE_NAMES[z] for z in profile.new_zombies) or "复习已有僵尸"
                backup = "、".join(PLANT_NAMES[p] for p in profile.candidates[:5]) or "无"
                note = f"本关教学：{new} · {threats}\n{profile.lesson}\n候选：{backup}；保留戴夫锁卡，按固定顺序补满实际卡槽。"
                if mode == "fixed_shuffled":
                    note += "\n每局先确定同一套预设卡，再随机打乱可选卡顺序；戴夫锁卡保持原位。下方显示预设顺序。"
            else:
                note = ("先按僵尸预览安排针对卡；实际卡组以测试结果为准。"
                    if mode in ("adaptive", "adaptive_shuffled") else
                    "按左到右优先选取，保留戴夫锁卡；未解锁卡跳过，空位补默认卡。支持中文名称或编号。"
                    if mode == "custom" else "兼容原生固定卡配置，需要与可选卡槽完全匹配。")
                if mode == "adaptive_shuffled":
                    note += "\n确定针对卡组后随机打乱可移动卡槽；戴夫锁卡保持原位。"
            color = self.ui.CLR_TEXT_MUTED
            if self.busy:
                note = "预设卡组预览（不影响当前运行配置）\n" + note
        except (ValueError, StopIteration) as exc:
            shown, note, color = (), f"请检查卡组：{exc}", self.ui.COLOR_RED
            is_special = False

        for widget in self.chips.winfo_children():
            widget.destroy()

        if is_special:
            # 传送带与趣味关卡无需选卡，不显示常规植物卡片，直接展示特殊说明横幅
            special_frame = tk.Frame(self.chips, bg="#111a25", highlightbackground="#2c4057", highlightthickness=1)
            special_frame.pack(fill=tk.X, padx=2, pady=4)
            tk.Label(
                special_frame,
                text=f"🎮 本关为特殊关卡【{profile.special}】，无需选卡",
                bg="#111a25", fg=self.ui.COLOR_BLUE, font=self.ui.FONT_BOLD,
                anchor="w", padx=10, pady=4,
            ).pack(fill=tk.X, pady=(6, 2))
            tk.Label(
                special_frame,
                text="本关植物由传送带自动传送或按特殊关卡规则掉落，游戏内无选卡界面，无需配置卡组。",
                bg="#111a25", fg=self.ui.CLR_TEXT_MUTED, font=self.ui.FONT_MAIN,
                anchor="w", padx=10, pady=4,
            ).pack(fill=tk.X, pady=(2, 6))
            self.mode.config(state="disabled")
            self.test_button.set_disabled(True)
            self.reset_button.set_disabled(True)
            self.edit_button.pack_forget()
            self.entry_row.pack_forget()
        else:
            for index, plant in enumerate(shown):
                chip = tk.Frame(self.chips, bg="#182333", highlightbackground="#2c4057", highlightthickness=1)
                chip.grid(row=index//4, column=index%4, sticky="ew", padx=3, pady=3)
                tk.Label(chip, text=f"{index+1:02d}  {PLANT_NAMES[plant]}", bg="#182333",
                         fg=self.ui.CLR_TEXT_MAIN, font=self.ui.FONT_MAIN, anchor="w",
                         padx=7, pady=7).pack(fill=tk.X)
            self.mode.config(state="disabled" if self.busy else "readonly")
            self.test_button.set_disabled(self.busy)
            self.reset_button.set_disabled(self.busy)

            # Legacy custom/native decks remain loadable.
            legacy_mode = self.mode_var.get() in (MODE_LABELS["custom"], MODE_LABELS["native"])
            if legacy_mode:
                self.edit_button.pack(side=tk.LEFT, padx=3, before=self.reset_button)
                self.entry_row.pack(fill=tk.X, padx=12, pady=(8, 0), before=self.note)
            else:
                self.edit_button.pack_forget()
                self.entry_row.pack_forget()

        self.note.config(text=note, fg=color)
        self.entry.config(state="disabled" if self.busy or is_special or self.mode_var.get() in
                          (MODE_LABELS[key] for key in VISIBLE_MODES) else "normal")

    def reset(self):
        self.cards_var.set("")
        self.mode_var.set(MODE_LABELS["fixed"])

    def custom_profiles(self):
        """Snapshot all per-level overrides on the Tk thread for batch LIVE runs."""
        self.refresh()
        return {key: parse_cards(cards) for key, (mode, cards) in self.profiles.items()
                if mode == MODE_LABELS["custom"]}

    def selection_profiles(self):
        self.refresh()
        modes = {label: key for key, label in MODE_LABELS.items()}
        return {key: (modes[mode], parse_cards(cards) if modes[mode] in ("custom", "native") else None)
                for key, (mode, cards) in self.profiles.items()}

    def set_busy(self, busy):
        if busy and not self._preview_initialized:
            level, playthrough = self.context()
            self._syncing = True
            try:
                self.preview_level_var.set(f"{(level-1)//10+1}-{(level-1)%10+1}")
                self.preview_playthrough_var.set(str(playthrough))
            finally:
                self._syncing = False
            self._preview_initialized = True
        self.busy = busy
        if busy:
            self.preview_row.pack(fill=tk.X, padx=12, pady=(0, 8), before=self.chips)
        else:
            self.preview_row.pack_forget()
        self.mode.config(state="disabled" if busy else "readonly")
        for button in (self.edit_button, self.reset_button, self.test_button):
            button.set_disabled(busy)
        self.refresh()

    def show_result(self, text, failed=False):
        self.note.config(text=text, fg=self.ui.COLOR_RED if failed else self.ui.COLOR_GREEN)

    def edit(self):
        if self.busy:
            return
        level, playthrough = self.context()
        try:
            selected = list(parse_cards(self.cards_var.get()) or default_deck_priority(level, playthrough))
        except ValueError:
            selected = list(default_deck_priority(level, playthrough))
        dialog = tk.Toplevel(self)
        dialog.title("编辑测试卡组")
        dialog.configure(bg=self.ui.CLR_PANEL)
        dialog.resizable(False, False)
        dialog.transient(self.winfo_toplevel())
        tk.Label(dialog, text="点选卡牌 · 再次点击移除", bg=self.ui.CLR_PANEL,
                 fg=self.ui.CLR_TEXT_MAIN, font=self.ui.FONT_TITLE).pack(anchor="w", padx=16, pady=(14, 4))
        summary = tk.Label(dialog, bg=self.ui.CLR_PANEL, fg=self.ui.COLOR_BLUE,
                           font=self.ui.FONT_MAIN, wraplength=650, justify=tk.LEFT)
        summary.pack(fill=tk.X, padx=16, pady=(0, 8))
        grid = tk.Frame(dialog, bg=self.ui.CLR_PANEL)
        grid.pack(padx=12)
        buttons = {}
        def update():
            summary.config(text=f"已选 {len(selected)} 张 · " + " → ".join(PLANT_NAMES[i] for i in selected))
            for plant, button in buttons.items():
                active = plant in selected
                button.configure(bg="#1c3853" if active else self.ui.CLR_CARD,
                                 fg=self.ui.COLOR_BLUE if active else self.ui.CLR_TEXT_MAIN,
                                 text=f"{selected.index(plant)+1:02d} · {PLANT_NAMES[plant]}" if active else PLANT_NAMES[plant])
        def toggle(plant):
            if plant in selected:
                selected.remove(plant)
            else:
                selected.append(plant)
            update()
        for plant, name in enumerate(PLANT_NAMES):
            button = tk.Button(grid, text=name, command=lambda i=plant: toggle(i),
                               font=self.ui.FONT_MAIN, relief=tk.FLAT, width=15, pady=8,
                               activebackground="#274461", activeforeground="white")
            button.grid(row=plant//5, column=plant%5, padx=3, pady=3)
            buttons[plant] = button
        bottom = tk.Frame(dialog, bg=self.ui.CLR_PANEL)
        bottom.pack(fill=tk.X, padx=16, pady=12)
        def apply():
            if not selected:
                summary.config(text="至少选择一张卡牌")
                return
            self.cards_var.set(",".join(map(str, selected)))
            self.mode_var.set(MODE_LABELS["custom"])
            dialog.destroy()
        ttk.Button(bottom, text="应用卡组", command=apply).pack(side=tk.RIGHT)
        ttk.Button(bottom, text="取消", command=dialog.destroy).pack(side=tk.RIGHT, padx=8)
        update()
        dialog.update_idletasks()
        owner = self.winfo_toplevel()
        x = max(0, owner.winfo_rootx() + (owner.winfo_width() - dialog.winfo_reqwidth()) // 2)
        y = max(0, owner.winfo_rooty() + (owner.winfo_height() - dialog.winfo_reqheight()) // 2)
        dialog.geometry(f"+{x}+{y}")
        dialog.grab_set()


def start_deck_test(gui, panel):
    if any(getattr(gui, name, False) for name in
           ("_live_running", "_training_running", "_static_test_running", "_deck_testing")):
        gui.log("WARN", "[SEED] 请先停止当前测试或训练")
        return
    try:
        mode, cards = panel.settings()
        level, playthrough = panel.context()
        profile = deck_profile(level, playthrough)
        if profile.special:
            panel.show_result(f"本关为特殊关卡【{profile.special}】，系统自动发牌，无需测试选卡。", False)
            gui.log("INFO", f"[SEED] 本关为特殊关卡【{profile.special}】，系统自动发牌，无需选卡")
            return
    except ValueError as exc:
        panel.show_result(str(exc), True)
        return
    gui._deck_testing = True
    gui._static_test_running = True
    panels = [p for p in (getattr(gui, "live_deck_panel", None), getattr(gui, "training_deck_panel", None)) if p is not None]
    for item in panels:
        item.set_busy(True)
    context_widgets = [getattr(gui, name) for name in
                       ("cmb_live_playthrough", "cmb_training_level", "cmb_training_playthrough")
                       if hasattr(gui, name)]
    for widget in context_widgets:
        widget.config(state="disabled")
    panel.show_result("正在准备关卡并测试选卡…")
    def finish(text, failed):
        gui._deck_testing = gui._static_test_running = False
        for item in panels:
            item.set_busy(False)
        for widget in context_widgets:
            widget.config(state="readonly")
        panel.show_result(text, failed)
        gui.log("FAIL" if failed else "PASS", f"[SEED] {text}")
    def worker():
        backend = None
        try:
            from pvz_gym.client import PvZClient
            backend = PvZClient(getattr(gui.client, "dll_path", None),
                                auto_select_seeds=False, auto_collect=False, auto_whack=False)
            backend.reset({"level_id": level, "playthrough": playthrough,
                           "stop_at_seed_selection": True, "auto_select_seeds": False})
            state = backend.get_seed_selection()
            if not state.present:
                text = "本关无交互选卡界面，跳过选卡；没有运行战斗策略。"
            else:
                logs = []
                if mode == "native" and cards:
                    # The test previews the same requested cards; actual fixed mode
                    # still requires an exact-size deck at native training reset.
                    if len(cards) != state.required_count:
                        raise ValueError(f"原生固定卡需要 {state.required_count} 张，当前 {len(cards)} 张")
                    if any(plant not in cards for plant in state.selected_seeds):
                        raise ValueError("固定卡与当前已选卡不匹配；请改用自定义优先级")
                    options = {opt.plant_type: opt for opt in state.options}
                    if any(plant not in options or not options[plant].unlocked
                           or (not options[plant].available and not options[plant].selected)
                           for plant in cards):
                        raise ValueError("固定卡包含当前不可选卡牌")
                ready = select_default_deck(backend, level, playthrough, confirm=False,
                                            priority_override=cards, report=logs.append,
                                            selection_mode=mode)
                names = [PLANT_NAMES[i] if i < 40 else f"植物{i}" for i in ready.selected_seeds]
                gaps = [line for line in logs if "缺口" in line or "补位提示" in line]
                text = f"已选 {ready.selected_count}/{ready.required_count}：" + "、".join(names)
                text += "。停留在选卡界面。" + ("\n" + "；".join(gaps) if gaps else "")
            failed = False
        except Exception as exc:
            text, failed = f"选卡测试失败：{exc}", True
        finally:
            if backend is not None:
                try:
                    backend.close()
                except Exception as exc:
                    text, failed = f"选卡测试清理失败：{exc}", True
        gui.root.after(0, finish, text, failed)
    threading.Thread(target=worker, daemon=True).start()
