# -*- coding: utf-8 -*-
"""Independent game-runtime and sampler controls for PvZ Gym training."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog
from pvz_gym.runtime import RuntimeConfig, TurboConfig
from pvz_gym.env_file import get_game_executable, set_game_executable
from train.runtime import TrainingRuntimeConfig
from .q_heatmap_overlay import get_q_heatmap_overlay

_SIMULATION_OPTIONS = (
    "Turbo (最快，全程无头)",
    "Headless (Fast)",
    "Continuous (No Pause)",
    "Stepped",
)
_DEFAULT_SIMULATION_LABEL = "Turbo (最快，全程无头)"

_Q_OVERLAY_OPTIONS = (
    "关闭",
    "Top 5 热力图",
)
_DEFAULT_Q_OVERLAY_LABEL = "关闭"

FIELDS = (
    ('num_envs', '环境数量', '6', ('1', '2', '4', '6', '8')),
    ('executor', '游戏执行器', 'legacy', ('legacy', 'resident')),
    ('transport', '游戏数据传输', 'json', ('json', 'binary', 'shared_memory')),
    ('snapshot', '观测复用', 'legacy', ('legacy', 'shared')),
    ('game_executable', '游戏程序路径（单开/多开）', '', None),
    ('process_ids', '指定 PID，逗号分隔', '', None),
    ('inference_batch_size', '推理批量', '6', ('1', '2', '4', '6', '8')),
    ('inference_max_wait_ms', '批量等待上限 ms', '2', None),
    ('queue_capacity', '采样队列容量', '1024', None),
    ('worker_timeout_s', '环境回复超时 s', '60', None),
    ('restart_limit', '单环境重启上限', '2', None),
    ('max_policy_lag', '策略落后更新上限', '64', None),
    ('ui_update_interval_ms', '训练显示间隔 ms', '200', None),
)

BOOLEANS = (
    ('fast_intro', 'Turbo 快速开场', True),
    ('fast_terminal', 'Turbo 快速结算', True),
    ('mute_audio', 'Turbo 静音', True),
    ('asynchronous', '异步采样', True),
    ('asynchronous_checkpoints', '异步保存', True),
    ('game_topmost', '游戏窗口置顶', False),
)


def _selected_simulation_mode(self) -> str:
    variable = getattr(self, "var_training_simulation_mode", None)
    value = str(variable.get()) if variable is not None else _DEFAULT_SIMULATION_LABEL
    normalized = value.strip().lower()
    if normalized.startswith("turbo"):
        return "turbo"
    if normalized.startswith("headless"):
        return "headless"
    if normalized.startswith("continuous"):
        return "continuous"
    return "stepped"


def build_runtime_controls(self, ui, first: int = 0):
    """构建 PvZ 专属运行与仿真参数面板（暗黑 Cyber 风格与逻辑分类）。"""
    target_frame = getattr(self, "training_runtime_frame", None)
    if target_frame is None:
        target_frame = getattr(self, "training_params_frame")

    # 清除旧组件，避免切换 Trainer 时重复堆叠
    for child in target_frame.winfo_children():
        child.destroy()

    for col in range(3):
        target_frame.columnconfigure(col, weight=1, uniform="runtime-parameter")

    values = getattr(self, '_runtime_param_vars', {})
    configured_executable = get_game_executable()
    widgets = []

    def _field_box(parent, row: int, col: int, label: str):
        box = tk.Frame(
            parent, bg=ui.CLR_PANEL,
            highlightbackground=ui.CLR_BORDER, highlightthickness=1,
        )
        box.grid(
            row=row, column=col, sticky="nsew",
            padx=(0 if col == 0 else 4, 0 if col == 2 else 4),
            pady=(0 if row == 0 else 8, 0),
        )
        field_label = tk.Label(
            box, text=label, font=("Microsoft YaHei UI", 8, "bold"),
            fg=ui.CLR_TEXT_MUTED, bg=ui.CLR_PANEL,
            anchor="w", justify=tk.LEFT, wraplength=180,
        )
        field_label.pack(fill=tk.X, padx=9, pady=(8, 0))
        field_label.bind(
            "<Configure>",
            lambda event, lbl=field_label: lbl.configure(wraplength=max(1, event.width)),
        )
        return box

    def _create_entry(box, variable: tk.StringVar, width: int = 15):
        entry = tk.Entry(
            box, textvariable=variable, width=width, font=ui.FONT_MONO,
            fg=ui.CLR_TEXT_MAIN, bg=ui.CLR_PANEL,
            insertbackground=ui.CLR_TEXT_MAIN, relief="flat",
            highlightthickness=1, highlightbackground=ui.CLR_BORDER,
            highlightcolor=ui.COLOR_BLUE,
            disabledbackground=ui.CLR_PANEL,
            disabledforeground=ui.CLR_TEXT_MUTED,
        )
        entry.pack(fill=tk.X, padx=9, pady=(4, 9), ipady=4)
        return entry

    def _create_combobox(box, variable: tk.StringVar, choices: tuple[str, ...], width: int = 15):
        cmb = ttk.Combobox(
            box, textvariable=variable, values=choices,
            state="readonly", width=width,
        )
        cmb.pack(fill=tk.X, padx=9, pady=(4, 9), ipady=2)
        return cmb

    # ---------------- 1. 核心仿真与交互模式 ----------------
    if not hasattr(self, "var_training_simulation_mode"):
        self.var_training_simulation_mode = tk.StringVar(value=_DEFAULT_SIMULATION_LABEL)
    if not hasattr(self, "var_training_q_overlay"):
        self.var_training_q_overlay = tk.StringVar(value=_DEFAULT_Q_OVERLAY_LABEL)

    box_sim = _field_box(target_frame, 0, 0, "Simulation (推进模式)")
    cmb_sim = _create_combobox(box_sim, self.var_training_simulation_mode, _SIMULATION_OPTIONS)
    self.cmb_training_simulation = cmb_sim
    preset = ui.CyberButton(box_sim, "单开可视化训练", lambda: _use_visible_preset(),
                           bg_color=ui.CLR_CARD, fg_color=ui.COLOR_BLUE,
                           height=26, padding_x=8, font=("Microsoft YaHei UI", 8))
    preset.pack(fill=tk.X, padx=9, pady=(0, 9))
    widgets.append((preset, False))

    box_q = _field_box(target_frame, 0, 1, "Q值热力图显示")
    cmb_q = _create_combobox(box_q, self.var_training_q_overlay, _Q_OVERLAY_OPTIONS)
    self.cmb_training_q_overlay = cmb_q

    def _get_overlay():
        try:
            from . import dqn_headless
            if hasattr(dqn_headless, "get_q_heatmap_overlay"):
                return dqn_headless.get_q_heatmap_overlay()
        except Exception:
            pass
        return get_q_heatmap_overlay()

    def _on_simulation_selected(_event=None):
        mode = _selected_simulation_mode(self)
        if mode == "turbo":
            for name, value in (('executor', 'resident'), ('transport', 'binary'), ('snapshot', 'shared')):
                if name in values:
                    values[name].set(value)
        if self.cmb_training_q_overlay is not None:
            if mode == "continuous":
                self.cmb_training_q_overlay.config(state="readonly")
                if hasattr(self, "var_training_q_overlay"):
                    self.var_training_q_overlay.set("Top 5 热力图")
                _get_overlay().set_enabled(True)
            else:
                self.cmb_training_q_overlay.config(state="disabled")
                if hasattr(self, "var_training_q_overlay"):
                    self.var_training_q_overlay.set(_DEFAULT_Q_OVERLAY_LABEL)
                _get_overlay().set_enabled(False)
                _get_overlay().clear()

    cmb_sim.bind("<<ComboboxSelected>>", _on_simulation_selected, add="+")

    def _on_q_overlay_selected(_event=None):
        val = str(self.var_training_q_overlay.get()) if hasattr(self, "var_training_q_overlay") else _DEFAULT_Q_OVERLAY_LABEL
        if val == "Top 5 热力图" and _selected_simulation_mode(self) == "continuous":
            _get_overlay().set_enabled(True)
        else:
            _get_overlay().set_enabled(False)
        controller = getattr(self, "_training_controller", None)
        if controller is not None and getattr(self, "_training_running", False):
            controller.set_visualization_enabled(
                val == "Top 5 热力图" and _selected_simulation_mode(self) == "continuous")
        if getattr(self, "_training_evaluating", False):
            self._training_eval_visualize = (
                val == "Top 5 热力图" and _selected_simulation_mode(self) == "continuous")

    def _use_visible_preset():
        self.var_training_simulation_mode.set("Continuous (No Pause)")
        values['num_envs'].set('1')
        values['asynchronous'].set(False)
        values['inference_batch_size'].set('1')
        existing_ids = [value.strip() for value in values['process_ids'].get().split(',') if value.strip()]
        if len(existing_ids) > 1:
            values['process_ids'].set(existing_ids[0])
        _on_simulation_selected()
        _on_q_overlay_selected()

    self._use_visible_training_preset = _use_visible_preset

    cmb_q.bind("<<ComboboxSelected>>", _on_q_overlay_selected, add="+")
    try:
        self.var_training_q_overlay.trace_add("write", lambda *_: _on_q_overlay_selected())
    except Exception:
        pass

    # 初始 Q 值控件状态
    if _selected_simulation_mode(self) != "continuous":
        cmb_q.config(state="disabled")

    from .parameter_sections import ParameterSection
    self.training_runtime_advanced = ParameterSection(target_frame, ui, "底层工程参数")
    self.training_runtime_advanced.grid(row=2, column=0, columnspan=3,
                                        sticky="ew", pady=(10, 0))
    advanced_frame = self.training_runtime_advanced.body
    seed_box = _field_box(advanced_frame, 0, 0, "随机种子")
    self.ent_training_seed = _create_entry(seed_box, self.var_training_seed)
    self.training_runtime_advanced.watch(self.var_training_seed, "0")

    # ---------------- 2. 字段参数排布 ----------------
    advanced_positions = {
        name: divmod(index + 1, 3)
        for index, (name, *_rest) in enumerate(
            field for field in FIELDS if field[0] != "num_envs"
        )
    }

    for name, label, default, choices in FIELDS:
        initial_val = configured_executable if name == 'game_executable' else default
        if name == 'executor' and configured_executable:
            initial_val = 'resident'
        if _selected_simulation_mode(self) == 'turbo':
            initial_val = {'executor': 'resident', 'transport': 'binary', 'snapshot': 'shared'}.get(name, initial_val)
        variable = values.setdefault(name, tk.StringVar(value=initial_val))
        if name == 'num_envs':
            row, col, parent = 0, 2, target_frame
        else:
            row, col = advanced_positions[name]
            parent = advanced_frame
            self.training_runtime_advanced.watch(variable, initial_val)
        box = _field_box(parent, row, col, label)

        if name == 'game_executable':
            path_row = tk.Frame(box, bg=ui.CLR_PANEL)
            path_row.pack(fill=tk.X, padx=9, pady=(4, 9))
            entry = tk.Entry(
                path_row, textvariable=variable, width=10, font=ui.FONT_MONO,
                fg=ui.CLR_TEXT_MAIN, bg=ui.CLR_PANEL,
                insertbackground=ui.CLR_TEXT_MAIN, relief="flat",
                highlightthickness=1, highlightbackground=ui.CLR_BORDER,
                highlightcolor=ui.COLOR_BLUE,
                disabledbackground=ui.CLR_PANEL,
                disabledforeground=ui.CLR_TEXT_MUTED,
            )
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)
            widgets.append((entry, False))

            def browse(var=variable):
                path = filedialog.askopenfilename(title='选择训练游戏程序', filetypes=[('游戏程序', '*.exe')])
                if path:
                    var.set(path)
                    try:
                        set_game_executable(path)
                    except Exception:
                        pass
                    if 'executor' in values:
                        values['executor'].set('resident')

            btn_browse = ui.CyberButton(
                path_row, "浏览", browse,
                bg_color=ui.CLR_CARD, fg_color=ui.COLOR_BLUE,
                height=26, padding_x=8, font=("Microsoft YaHei UI", 8),
            )
            btn_browse.pack(side=tk.RIGHT, padx=(6, 0))
            widgets.append((btn_browse, False))
            from .game_setup import prepare_from_gui
            btn_prepare = ui.CyberButton(
                box, "准备训练存档", lambda: prepare_from_gui(self),
                bg_color=ui.CLR_CARD, fg_color=ui.COLOR_BLUE,
                height=26, padding_x=8, font=("Microsoft YaHei UI", 8))
            btn_prepare.pack(fill=tk.X, padx=9, pady=(0, 9))
            widgets.append((btn_prepare, False))
        elif choices:
            cmb = _create_combobox(box, variable, choices)
            widgets.append((cmb, True))
            if name == 'num_envs':
                def choose_count(_event=None):
                    try:
                        count = int(values['num_envs'].get())
                    except ValueError:
                        count = 1
                    values['asynchronous'].set(count > 1)
                    values['inference_batch_size'].set(str(count))
                    if count > 1:
                        self.var_training_q_overlay.set(_DEFAULT_Q_OVERLAY_LABEL)
                        self.cmb_training_q_overlay.config(state='disabled')
                    elif _selected_simulation_mode(self) == 'continuous':
                        self.cmb_training_q_overlay.config(state='readonly')
                cmb.bind('<<ComboboxSelected>>', choose_count)
        else:
            entry = _create_entry(box, variable)
            widgets.append((entry, False))

    # ---------------- 3. Turbo 优化与异步选项（独立深色开关栏） ----------------
    toggles_container = tk.Frame(
        advanced_frame, bg=ui.CLR_PANEL,
        highlightbackground=ui.CLR_BORDER, highlightthickness=1,
    )
    toggles_container.grid(
        row=5, column=0, columnspan=3, sticky="nsew",
        padx=0, pady=(8, 0),
    )
    for c in range(3):
        toggles_container.columnconfigure(c, weight=1, uniform="toggle-col")

    for idx, (name, label, default) in enumerate(BOOLEANS):
        variable = values.setdefault(name, tk.BooleanVar(value=default))
        self.training_runtime_advanced.watch(variable, default)
        toggle_box = tk.Frame(toggles_container, bg=ui.CLR_PANEL)
        toggle_box.grid(row=idx // 3, column=idx % 3, sticky="ew", padx=4, pady=6)

        chk = tk.Checkbutton(
            toggle_box,
            text=label,
            variable=variable,
            font=("Microsoft YaHei UI", 8, "bold"),
            fg=ui.CLR_TEXT_MAIN,
            bg=ui.CLR_PANEL,
            activebackground=ui.CLR_PANEL,
            activeforeground=ui.COLOR_BLUE,
            selectcolor=ui.CLR_CARD,
            bd=0,
            highlightthickness=0,
            cursor="hand2",
            padx=2,
            pady=2,
        )
        chk.pack(anchor="w")
        widgets.append((chk, False))

    tk.Label(target_frame,
             text="单开可视化：1 个环境 + Continuous。Pause 确认后可 Reset Game，再 Continue；热力图开关保留模型与 Replay。",
             font=("Microsoft YaHei UI", 8), fg=ui.CLR_TEXT_MUTED, bg=ui.CLR_PANEL,
             anchor="w", justify=tk.LEFT, wraplength=570).grid(
                 row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))

    self._runtime_param_vars = values
    self._runtime_widgets = widgets


def runtime_settings(self, *, env=None, runtime=None):
    values = getattr(self, '_runtime_param_vars', {})
    if not values:
        return env.runtime if env else RuntimeConfig(), runtime or TrainingRuntimeConfig(), env.turbo if env else TurboConfig()
    ids = tuple(int(v.strip()) for v in values['process_ids'].get().split(',') if v.strip())
    executable = values['game_executable'].get().strip() or get_game_executable() or None
    if executable:
        from pvz_gym.env_file import resolve_game_paths
        executable = str(resolve_game_paths(executable).executable.resolve(strict=True))
    base = runtime or TrainingRuntimeConfig()
    training = replace(
        base,
        num_envs=int(values['num_envs'].get()),
        process_ids=ids,
        game_executable=executable,
        asynchronous=bool(values['asynchronous'].get()),
        asynchronous_checkpoints=bool(values['asynchronous_checkpoints'].get()),
        inference_batch_size=int(values['inference_batch_size'].get()),
        inference_max_wait_ms=float(values['inference_max_wait_ms'].get()),
        queue_capacity=int(values['queue_capacity'].get()),
        worker_timeout_s=float(values['worker_timeout_s'].get()),
        restart_limit=int(values['restart_limit'].get()),
        max_policy_lag=int(values['max_policy_lag'].get()),
        ui_update_interval_ms=int(values['ui_update_interval_ms'].get()),
        visualize_q=(_selected_simulation_mode(self) == 'continuous'
                     and str(self.var_training_q_overlay.get()) == 'Top 5 热力图'),
        game_topmost=bool(values['game_topmost'].get()),
    )
    if training.num_envs > 1 and not (ids or executable):
        raise ValueError('多开需要指定游戏路径或每个环境的 PID。')
    if training.visualize_q and training.num_envs != 1:
        raise ValueError('Q 热力图训练请使用 1 个环境，或关闭热力图继续多开训练。')
    game = RuntimeConfig(
        process_id=env.runtime.process_id if env else None,
        executor=values['executor'].get(),
        transport=values['transport'].get(),
        snapshot=values['snapshot'].get(),
    )
    turbo = replace(
        env.turbo if env else TurboConfig(),
        **{name: bool(values[name].get()) for name in ('fast_intro', 'fast_terminal', 'mute_audio')},
    )
    return game, training, turbo


def apply_runtime_settings(self, config):
    values = getattr(self, '_runtime_param_vars', {})
    for name, label, default, choices in FIELDS:
        if name not in values:
            continue
        value = getattr(config.env.runtime, name) if name in ('executor', 'transport', 'snapshot') else getattr(config.runtime, name)
        if name == 'process_ids':
            value = ','.join(map(str, value))
        values[name].set('' if value is None else str(value))
    for name, label, default in BOOLEANS:
        if name in values:
            values[name].set(
                getattr(config.env.turbo, name) if name in ('fast_intro', 'fast_terminal', 'mute_audio')
                else getattr(config.runtime, name)
            )
