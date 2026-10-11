# -*- coding: utf-8 -*-
"""Workbench smoke checks for the DQN training path.

The smoke control intentionally has two modes:
- no checkpoint loaded: run one or more real DQN epsilon=1 training-path games,
  including replay insertion and learning once replay warmup is available;
- checkpoint loaded: run one or more greedy evaluation games using that checkpoint.

Winning is not required. Every requested game must reach a clean WON/LOST terminal
state. Environment smoke also records detailed per-stage timings for the complete
DQN path so performance regressions remain visible.
"""

from __future__ import annotations

import threading
import time
import tkinter as tk
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from pvz_gym.client import PvZClient
from pvz_gym.env import PvZEnv
from train.dqn.agent import DoubleDQNAgent
from train.dqn.config import DoubleDQNConfig
from train.dqn.exploration import (
    DEFAULT_EXPLORATION_RATIO, EXPLORATION_MODE_LABELS, parse_action_family_weights, parse_exploration_ratio,
)
from train.dqn.credit import EpisodeEventCredit
from train.dqn.evaluate import EvaluationSummary, evaluate_checkpoint
from train.dqn.features import normalize_observation
from train.dqn.replay import ReplayBuffer
from .q_heatmap_overlay import get_q_heatmap_overlay
from .training_visualization import TrainingVisualizationBridge


@dataclass
class _EnvironmentSmokeSummary:
    requested_games: int
    completed_games: int = 0
    wins: int = 0
    losses: int = 0
    invalid_actions: int = 0
    unknown_terminals: int = 0
    stopped: bool = False
    episode_lengths: list[int] = field(default_factory=list)
    episode_rewards: list[float] = field(default_factory=list)
    total_ticks: int = 0
    total_steps: int = 0
    tick_samples: int = 0
    long_tick_jumps: int = 0
    wall_seconds: float = 0.0
    perf_lines: list[str] = field(default_factory=list)

    @property
    def terminal_games(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        return self.wins / self.terminal_games if self.terminal_games else 0.0

    @property
    def mean_steps(self) -> float:
        return float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0

    @property
    def mean_ticks_advanced(self) -> float:
        return self.total_ticks / self.tick_samples if self.tick_samples else 0.0

    @property
    def long_tick_jump_rate(self) -> float:
        return self.long_tick_jumps / self.tick_samples if self.tick_samples else 0.0


class _PerfStats:
    _ORDER = (
        "reset",
        "select_action",
        "env_step",
        "normalize_observation",
        "training_reward",
        "replay_add",
        "replay_sample",
        "learn",
    )

    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = defaultdict(list)

    def add_ms(self, name: str, started: float) -> None:
        self.samples[name].append((time.perf_counter() - started) * 1000.0)

    def lines(self) -> list[str]:
        lines: list[str] = []
        for name in self._ORDER:
            values = self.samples.get(name, ())
            if not values:
                continue
            array = np.asarray(values, dtype=np.float64)
            lines.append(
                f"[BENCH] {name:24s} n={array.size:5d} "
                f"avg={array.mean():8.3f} ms "
                f"p50={np.percentile(array, 50):8.3f} "
                f"p95={np.percentile(array, 95):8.3f} "
                f"max={array.max():8.3f}"
            )
        return lines


def _selected_simulation_mode(self) -> str:
    variable = getattr(self, "var_training_simulation_mode", None)
    text = str(variable.get()) if variable is not None else "Headless (Fast)"
    normalized = text.strip().lower()
    if normalized.startswith("turbo"):
        return "turbo"
    if normalized.startswith("continuous"):
        return "continuous"
    if normalized.startswith("stepped"):
        return "stepped"
    return "headless"


def _games_label(count: int) -> str:
    return f"{count} {'GAME' if count == 1 else 'GAMES'}"


def install_training_smoke_gui(gui_cls: Any, ui: Any) -> None:
    if getattr(gui_cls, "_training_smoke_gui_installed", False):
        return

    original_build_left_panel = gui_cls._build_left_panel
    original_set_checkpoint_controls = gui_cls._set_training_checkpoint_controls

    def _build_left_panel(self, parent):
        original_build_left_panel(self, parent)
        test_button = getattr(self, "btn_training_test_checkpoint", None)
        if test_button is None:
            return

        checkpoint_evaluation = test_button.master
        checkpoint_body = checkpoint_evaluation.master

        # Keep the Smoke row visually identical to the Test games row.
        existing_children = checkpoint_body.winfo_children()
        note = existing_children[-1] if existing_children else None
        self.var_training_smoke_games = tk.StringVar(value="1")
        smoke_row = tk.Frame(checkpoint_body, bg=ui.CLR_CARD)
        if note is not None:
            smoke_row.pack(fill=tk.X, pady=(7, 0), before=note)
        else:
            smoke_row.pack(fill=tk.X, pady=(7, 0))

        smoke_box = tk.Frame(smoke_row, bg=ui.CLR_CARD)
        smoke_box.pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(
            smoke_box,
            text="Smoke games",
            font=("Microsoft YaHei UI", 8, "bold"),
            fg=ui.CLR_TEXT_MUTED,
            bg=ui.CLR_CARD,
        ).pack(anchor="w")
        self.ent_training_smoke_games = tk.Entry(
            smoke_box,
            textvariable=self.var_training_smoke_games,
            width=7,
            font=ui.FONT_MONO,
            fg=ui.CLR_TEXT_MAIN,
            bg=ui.CLR_PANEL,
            insertbackground=ui.CLR_TEXT_MAIN,
            relief="flat",
            highlightthickness=1,
            highlightbackground=ui.CLR_BORDER,
            highlightcolor=ui.COLOR_BLUE,
            disabledbackground=ui.CLR_PANEL,
            disabledforeground=ui.CLR_TEXT_MUTED,
        )
        self.ent_training_smoke_games.pack(fill=tk.X, pady=(3, 0), ipady=4)

        self.btn_training_smoke = ui.CyberButton(
            smoke_row,
            "Smoke test",
            self._run_training_smoke,
            bg_color=ui.CLR_PANEL,
            fg_color=ui.COLOR_BLUE,
            height=32,
        )
        self.btn_training_smoke.pack(side=tk.LEFT, fill=tk.X, expand=True)

        try:
            is_dqn = self._selected_training_spec().key == "double_dqn"
        except Exception:
            is_dqn = False
        self._set_training_checkpoint_controls(is_dqn)

    def _set_training_checkpoint_controls(self, dqn: bool):
        original_set_checkpoint_controls(self, dqn)
        button = getattr(self, "btn_training_smoke", None)
        if button is not None:
            button.set_disabled((not dqn) or self._training_running)
        entry = getattr(self, "ent_training_smoke_games", None)
        if entry is not None:
            entry.configure(state=tk.DISABLED if ((not dqn) or self._training_running) else tk.NORMAL)

    def _run_training_smoke(self):
        if getattr(self, "_preparing_training_save", False):
            self.log("WARN", "[SMOKE] 等待训练存档准备完成。")
            return
        if getattr(self, "_training_running", False):
            return
        if getattr(self, "_live_running", False) or getattr(self, "_static_test_running", False):
            self.log("WARN", "[SMOKE] Adventure LIVE or Native Test is already running")
            return
        try:
            if self._selected_training_spec().key != "double_dqn":
                raise RuntimeError("Smoke test is currently wired to Double DQN only")
            level_id, ticks, playthrough, seed, mower_penalty, seed_types = (
                self._parse_common_training_config()
            )
            smoke_games = int(self.var_training_smoke_games.get().strip())
            if smoke_games <= 0:
                raise ValueError("Smoke games must be positive")
            from .training_runtime import runtime_settings
            game, runtime, turbo = runtime_settings(self)
            self._training_smoke_runtime_settings = (game, replace(runtime, num_envs=1, inference_batch_size=1), turbo)
            exploration_options = {}
            if not self._training_checkpoint_path:
                variables = self._training_param_vars
                exploration_mode = next(key for key, label in EXPLORATION_MODE_LABELS.items()
                                        if label == variables["exploration_mode"].get())
                ratio = (DEFAULT_EXPLORATION_RATIO if exploration_mode == "epsilon_random" else
                         parse_exploration_ratio(variables["exploration_ratio"].get()))
                exploration_options = {
                    "exploration_mode": exploration_mode,
                    "exploration_ratio": ratio,
                    "exploration_type_weights": parse_action_family_weights(
                        variables["exploration_type_weights"].get()),
                }
        except Exception as exc:
            self.lbl_training_progress.config(
                text=f"Smoke config error: {exc}",
                fg=ui.COLOR_RED,
            )
            self.log("FAIL", f"[SMOKE] config error: {exc}")
            return

        checkpoint = self._training_checkpoint_path
        mode = _selected_simulation_mode(self)
        if mode == "continuous":
            try:
                if hasattr(self, "var_training_q_overlay") and self.var_training_q_overlay.get() == "关闭":
                    self.var_training_q_overlay.set("Top 5 热力图")
                overlay_enabled = True
                if hasattr(self, "var_training_q_overlay"):
                    overlay_enabled = (str(self.var_training_q_overlay.get()) == "Top 5 热力图")
                get_q_heatmap_overlay().set_enabled(overlay_enabled)
            except Exception:
                pass
        self._training_running = True
        self._training_evaluating = True
        self._static_test_running = True
        self._training_controller = None
        self._training_pause_event = None
        self._training_stop_event = threading.Event()
        self._set_training_controls(True)
        self.btn_training_pause.set_disabled(True)
        self.btn_training_continue.set_disabled(True)
        self.lbl_training_status.config(text="SMOKE", fg=ui.COLOR_BLUE)

        games_text = _games_label(smoke_games)
        level_label = self.var_training_level.get()
        if checkpoint:
            self._training_run_token = uuid4().hex
            self._training_eval_visualize = (mode == "continuous"
                and (not hasattr(self, "var_training_q_overlay") or str(self.var_training_q_overlay.get()) == "Top 5 热力图"))
            bridge = getattr(self, "_training_visualization_bridge", None)
            if bridge is None:
                bridge = TrainingVisualizationBridge(self.root, overlay=get_q_heatmap_overlay())
                self._training_visualization_bridge = bridge
            bridge.start(self._training_run_token)
            self.lbl_training_progress.config(
                text=f"Smoke {smoke_games} game(s) · {level_label} · {mode} · loaded model {Path(checkpoint).name}...",
                fg=ui.COLOR_BLUE,
            )
            self.log(
                "HEADER",
                f"\n========== DQN MODEL SMOKE {Path(checkpoint).name} · {level_label} · {games_text} =========="
            )
            threading.Thread(
                target=self._model_smoke_worker,
                args=(checkpoint, level_id, ticks, playthrough, seed_types, smoke_games, mode),
                daemon=True,
            ).start()
        else:
            level_label = self.var_training_level.get()
            self.lbl_training_progress.config(
                text=f"Smoke {smoke_games} game(s) · DQN epsilon=1 · {mode}...",
                fg=ui.COLOR_BLUE,
            )
            self.log(
                "HEADER",
                f"\n========== DQN ENV SMOKE {level_label} · "
                f"{mode.upper()} · {games_text} =========="
            )
            threading.Thread(
                target=self._environment_smoke_worker,
                args=(
                    level_id,
                    ticks,
                    playthrough,
                    seed,
                    seed_types,
                    mode,
                    smoke_games,
                    mower_penalty,
                    exploration_options,
                ),
                daemon=True,
            ).start()

    def _environment_smoke_worker(
        self,
        level_id: int,
        ticks: int,
        playthrough: int,
        seed: int,
        seed_types,
        simulation_mode: str,
        smoke_games: int,
        mower_penalty: float,
        exploration_options: dict,
    ):
        env = instance = None
        try:
            config = DoubleDQNConfig(
                env=DoubleDQNConfig().env.__class__(
                    level_id=level_id,
                    playthrough=playthrough,
                    ticks_per_step=ticks,
                    simulation_mode=simulation_mode,
                    max_episode_ticks=self._training_max_episode_ticks,
                    seed_selection_mode=getattr(self, "_training_seed_selection_mode", "native"),
                    seed_priority=seed_types,
                ),
                total_steps=max(1, smoke_games * 5_000),
                epsilon_start=1.0,
                epsilon_end=1.0,
                epsilon_decay_steps=1,
                mower_penalty=mower_penalty,
                device="auto",
                **exploration_options,
            )
            config = replace(config, **getattr(self, "_training_reward_options", {}))
            settings = getattr(self, "_training_smoke_runtime_settings", None)
            if settings:
                game, runtime, turbo = settings
                config = replace(config, env=replace(config.env, runtime=game, turbo=turbo), runtime=runtime)
            from train.sampler import native_environment
            env, instance = native_environment(config, config.runtime, 0,
                getattr(self.client, "dll_path", None), seed_types, max_episode_steps=5_000)
            client = env.session.backend
            summary = _EnvironmentSmokeSummary(requested_games=smoke_games)
            perf = _PerfStats()
            smoke_started = time.perf_counter()
            if str(simulation_mode).lower().startswith("continuous"):
                try:
                    _q_overlay = get_q_heatmap_overlay()
                    _pid = None
                    try:
                        _pid = client.read_snapshot().identity.process_id
                    except Exception:
                        pass
                    if not _pid:
                        _pid = getattr(client, "process_id", None) or getattr(env, "runtime_process_id", None)
                    if _pid:
                        _q_overlay.bind_target(_pid)
                except Exception:
                    pass
            action_dim = int(env.action_space.n)
            agent = DoubleDQNAgent(action_dim, config, device="auto")
            replay = ReplayBuffer(config.replay_capacity, action_dim, seed=seed + 1,
                                  credit_assignment=config.credit_assignment)
            global_step = 0

            for game_index in range(smoke_games):
                if self._training_stop_event is not None and self._training_stop_event.is_set():
                    summary.stopped = True
                    break

                timing = time.perf_counter()
                observation, info = env.reset(seed=seed + game_index)
                credit = EpisodeEventCredit(replay, config, info)
                perf.add_ms("reset", timing)

                timing = time.perf_counter()
                observation = credit.observe(normalize_observation(observation))
                perf.add_ms("normalize_observation", timing)

                episode_reward = 0.0
                result = "UNKNOWN"
                steps = 0

                for steps in range(1, 5_001):
                    if self._training_stop_event is not None and self._training_stop_event.is_set():
                        summary.stopped = True
                        break

                    mask = np.asarray(info.get("action_mask", ()), dtype=np.bool_)
                    if mask.shape != (action_dim,):
                        raise RuntimeError(
                            f"action mask shape {mask.shape} != {(action_dim,)}"
                        )
                    if not mask.any():
                        raise RuntimeError("playing state exposed no legal actions")

                    timing = time.perf_counter()
                    action = agent.select_action(
                        observation,
                        mask,
                        env_step=global_step,
                        explore=True,
                    )
                    perf.add_ms("select_action", timing)

                    if str(simulation_mode).lower().startswith("continuous"):
                        try:
                            _q_overlay = get_q_heatmap_overlay()
                            if _q_overlay.is_enabled():
                                _vis_data = agent.extract_q_visualization_data(observation, mask, top_k=5)
                                _q_overlay.update_data(_vis_data["top5"], _vis_data["noop_q"], scene_id=_vis_data["scene_id"])
                        except Exception:
                            pass

                    timing = time.perf_counter()
                    next_observation, raw_reward, terminated, truncated, next_info = env.step(action)
                    perf.add_ms("env_step", timing)

                    timing = time.perf_counter()
                    next_observation = normalize_observation(next_observation)
                    perf.add_ms("normalize_observation", timing)

                    next_mask = np.asarray(next_info.get("action_mask", ()), dtype=np.bool_)
                    if next_mask.shape != (action_dim,):
                        raise RuntimeError(
                            f"next action mask shape {next_mask.shape} != {(action_dim,)}"
                        )

                    done = bool(terminated or truncated)
                    timing = time.perf_counter()
                    reward = credit.add(
                        replay, observation, action, raw_reward, next_observation,
                        terminated, truncated, next_mask, next_info,
                    )
                    perf.add_ms("replay_add", timing)

                    global_step += 1
                    summary.total_steps += 1
                    if len(replay) >= config.batch_size and global_step % config.train_every == 0:
                        timing = time.perf_counter()
                        batch = replay.sample(config.batch_size)
                        perf.add_ms("replay_sample", timing)
                        timing = time.perf_counter()
                        agent.learn(batch)
                        perf.add_ms("learn", timing)

                    action_result = next_info.get("action_result") or {}
                    if isinstance(action_result, dict) and not bool(
                        action_result.get("success", False)
                    ):
                        summary.invalid_actions += 1

                    ticks_requested = max(
                        1,
                        int(next_info.get("ticks_requested", ticks) or 1),
                    )
                    ticks_advanced = max(
                        0,
                        int(next_info.get("ticks_advanced", ticks_requested) or 0),
                    )
                    summary.total_ticks += ticks_advanced
                    summary.tick_samples += 1
                    if ticks_advanced > ticks_requested * 2:
                        summary.long_tick_jumps += 1

                    episode_reward += float(reward)
                    result = str(next_info.get("result", "UNKNOWN"))
                    observation, info = next_observation, next_info

                    if done:
                        if result == "WON":
                            summary.wins += 1
                        elif result == "LOST":
                            summary.losses += 1
                        else:
                            summary.unknown_terminals += 1
                        break
                else:
                    summary.unknown_terminals += 1

                if summary.stopped:
                    break
                summary.completed_games += 1
                summary.episode_lengths.append(steps)
                summary.episode_rewards.append(episode_reward)

            summary.wall_seconds = max(time.perf_counter() - smoke_started, 1e-9)
            summary.perf_lines = perf.lines()
            summary.perf_lines.extend(
                (
                    f"[BENCH] {'steps_per_second':24s} "
                    f"{summary.total_steps / summary.wall_seconds:12.2f}",
                    f"[BENCH] {'game_ticks_per_second':24s} "
                    f"{summary.total_ticks / summary.wall_seconds:12.2f}",
                    f"[BENCH] {'wall_seconds':24s} {summary.wall_seconds:12.3f}",
                )
            )
            result = summary
        except Exception as exc:
            result = exc
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            if instance is not None:
                instance.close()
        self.root.after(0, self._finish_training_smoke, "environment", result, None)

    def _model_smoke_worker(
        self,
        checkpoint: str,
        level_id: int,
        ticks: int,
        playthrough: int,
        seed_types,
        smoke_games: int,
        simulation_mode: str,
    ):
        run_token = getattr(self, "_training_run_token", None)
        def on_event(event, data):
            if event in ("visualization", "visualization_clear"):
                bridge = getattr(self, "_training_visualization_bridge", None)
                if bridge is not None:
                    bridge.post(run_token, event, data)
                return

        try:
            result = evaluate_checkpoint(
                checkpoint,
                level_id=level_id,
                ticks_per_step=ticks,
                playthrough=playthrough,
                max_episode_ticks=self._training_max_episode_ticks or 0,
                seed_count=smoke_games,
                device="auto",
                dll_path=getattr(self.client, "dll_path", None),
                seed_types=seed_types,
                seed_selection_mode=getattr(self, "_training_seed_selection_mode", "native"),
                simulation_mode=simulation_mode,
                stop_event=self._training_stop_event,
                on_event=on_event,
                visualize_q=lambda: bool(getattr(self, "_training_eval_visualize", False)),
                training_runtime=(getattr(self, "_training_smoke_runtime_settings", (None, None, None))[1]),
            )
        except Exception as exc:
            result = exc
        self.root.after(0, self._finish_training_smoke, "model", result, checkpoint)

    def _finish_training_smoke(self, source: str, result, checkpoint: str | None):
        bridge = getattr(self, "_training_visualization_bridge", None)
        if bridge is not None:
            bridge.close()
        self._training_run_token = None
        self._training_eval_visualize = False
        try:
            get_q_heatmap_overlay().clear()
            get_q_heatmap_overlay().bind_target(None)
        except Exception:
            pass
        self._training_running = False
        self._training_evaluating = False
        self._static_test_running = False
        self._training_pause_event = None
        self._training_stop_event = None
        self._set_training_controls(False)

        if isinstance(result, Exception):
            self.lbl_training_status.config(text="FAIL", fg=ui.COLOR_RED)
            self.lbl_training_progress.config(text=f"Smoke failed: {result}", fg=ui.COLOR_RED)
            self.log("FAIL", f"[SMOKE:{source.upper()}] {result}")
            self.refresh_status()
            return

        if source == "environment":
            summary: _EnvironmentSmokeSummary = result
            stopped = bool(summary.stopped)
            clean_terminal = (
                summary.completed_games == summary.requested_games
                and summary.terminal_games == summary.requested_games
                and summary.invalid_actions == 0
                and summary.unknown_terminals == 0
                and not stopped
            )
            self._set_training_metric_values(
                {
                    "step": str(int(round(summary.mean_steps))),
                    "episode": f"{summary.completed_games}/{summary.requested_games}",
                    "record": f"{summary.wins} / {summary.losses}",
                    "win_rate": f"{summary.win_rate:.1%}",
                    "reward": f"{summary.mean_reward:.3f}",
                    "loss": "-",
                    "epsilon": "1.0",
                    "replay": "-",
                    "ticks": f"{summary.mean_ticks_advanced:.1f}",
                    "long_ticks": (
                        f"{summary.long_tick_jumps} "
                        f"({summary.long_tick_jump_rate:.1%})"
                    ),
                    "invalid": str(summary.invalid_actions),
                }
            )
            for line in summary.perf_lines:
                self.log("INFO", line)
            detail = (
                f"DQN epsilon=1 · {summary.wins}W/{summary.losses}L · "
                f"mean steps {summary.mean_steps:.0f}"
            )
        else:
            summary: EvaluationSummary = result
            requested = int(self.var_training_smoke_games.get().strip())
            stopped = bool(summary.stopped)
            clean_terminal = (
                summary.episodes == requested
                and summary.wins + summary.losses == requested
                and summary.unknown == 0
                and not stopped
            )
            self._set_training_metric_values(
                {
                    "step": str(int(round(summary.mean_steps))),
                    "episode": f"{summary.episodes}/{requested}",
                    "record": f"{summary.wins} / {summary.losses}",
                    "win_rate": f"{summary.win_rate:.1%}",
                    "reward": f"{summary.mean_score:.3f}",
                    "loss": "-",
                    "epsilon": "0 (greedy)",
                    "replay": "-",
                    "ticks": f"{summary.mean_ticks_advanced:.1f} ± {summary.std_ticks_advanced:.1f}",
                    "long_ticks": (
                        f"{summary.long_tick_jumps} ({summary.long_tick_jump_rate:.1%})"
                    ),
                }
            )
            name = Path(checkpoint).name if checkpoint else "checkpoint"
            detail = (
                f"{name} · {summary.wins}W/{summary.losses}L · "
                f"mean steps {summary.mean_steps:.0f} · score {summary.mean_score:.3f}"
            )

        if stopped:
            status = "STOPPED"
            color = ui.CLR_TEXT_MUTED
            log_level = "WARN"
        elif clean_terminal:
            status = "PASS"
            color = ui.COLOR_GREEN
            log_level = "PASS"
        else:
            status = "FAIL"
            color = ui.COLOR_RED
            log_level = "FAIL"

        self.lbl_training_status.config(text=status, fg=color)
        self.lbl_training_progress.config(text=f"Smoke {status} · {detail}", fg=color)
        self.log(log_level, f"[SMOKE:{source.upper()}] {status} · {detail}")
        self.refresh_status()

    gui_cls._run_training_smoke = _run_training_smoke
    gui_cls._environment_smoke_worker = _environment_smoke_worker
    gui_cls._model_smoke_worker = _model_smoke_worker
    gui_cls._finish_training_smoke = _finish_training_smoke
    gui_cls._set_training_checkpoint_controls = _set_training_checkpoint_controls
    gui_cls._build_left_panel = _build_left_panel
    gui_cls._training_smoke_gui_installed = True


__all__ = ["install_training_smoke_gui"]
