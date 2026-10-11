"""Controllable Double DQN training core shared by CLI and Workbench."""

from __future__ import annotations

from pvz_gym.profiling import TimingRecorder, bridge_timing
from time import perf_counter

import json
import logging
import math
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from pvz_gym.client import PvZClient
from pvz_gym.env import PvZEnv
from pvz_gym.runtime import require_runtime_implementation
from train.runtime import uses_parallel_runtime

from .agent import DoubleDQNAgent
from .config import DoubleDQNConfig, validate_resume_config
from .features import normalize_observation
from .credit import EpisodeEventCredit
from .reward import mower_event_count, effective_reward_settings
from .replay import ReplayBuffer


EventCallback = Callable[[str, dict[str, Any]], None]
logger = logging.getLogger(__name__)

_DEFAULT_EVAL_EVERY_EPISODES = 20
_DEFAULT_EVAL_SEED_START = 10_001
_DEFAULT_EVAL_SEED_COUNT = 20
_DEFAULT_EVAL_MAX_EPISODE_STEPS = None


@dataclass
class DoubleDQNSummary:
    requested_steps: int
    env_steps: int = 0
    episodes: int = 0
    wins: int = 0
    losses: int = 0
    invalid_actions: int = 0
    mowers_used: int = 0
    raw_episode_rewards: list[float] = field(default_factory=list)
    training_episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    last_loss: float | None = None
    replay_size: int = 0
    replay_restored: bool = False
    epsilon: float = 1.0
    stopped: bool = False
    checkpoint: str = ""
    tick_samples: int = 0
    ticks_advanced_sum: int = 0
    ticks_advanced_sq_sum: float = 0.0
    long_tick_jumps: int = 0
    eval_runs: int = 0
    best_eval_win_rate: float = 0.0
    best_eval_mean_score: float = 0.0
    best_checkpoint: str = ""

    @property
    def terminal_episodes(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        return self.wins / self.terminal_episodes if self.terminal_episodes else 0.0

    @property
    def mean_training_reward(self) -> float:
        return float(np.mean(self.training_episode_rewards)) if self.training_episode_rewards else 0.0

    @property
    def mean_episode_length(self) -> float:
        return float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0

    @property
    def mean_ticks_advanced(self) -> float:
        return self.ticks_advanced_sum / self.tick_samples if self.tick_samples else 0.0

    @property
    def std_ticks_advanced(self) -> float:
        if not self.tick_samples:
            return 0.0
        variance = self.ticks_advanced_sq_sum / self.tick_samples - self.mean_ticks_advanced**2
        return math.sqrt(max(0.0, variance))

    @property
    def long_tick_jump_rate(self) -> float:
        return self.long_tick_jumps / self.tick_samples if self.tick_samples else 0.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "requested_steps": self.requested_steps,
            "env_steps": self.env_steps,
            "episodes": self.episodes,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "invalid_actions": self.invalid_actions,
            "mowers_used": self.mowers_used,
            "last_loss": self.last_loss,
            "replay_size": self.replay_size,
            "replay_restored": self.replay_restored,
            "epsilon": self.epsilon,
            "mean_training_reward": self.mean_training_reward,
            "mean_episode_length": self.mean_episode_length,
            "tick_samples": self.tick_samples,
            "mean_ticks_advanced": self.mean_ticks_advanced,
            "std_ticks_advanced": self.std_ticks_advanced,
            "long_tick_jumps": self.long_tick_jumps,
            "long_tick_jump_rate": self.long_tick_jump_rate,
            "eval_runs": self.eval_runs,
            "best_eval_win_rate": self.best_eval_win_rate,
            "best_eval_mean_score": self.best_eval_mean_score,
            "best_checkpoint": self.best_checkpoint,
            "stopped": self.stopped,
            "checkpoint": self.checkpoint,
        }


def _episode_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**32, dtype=np.uint64))


def _require_mask(info: dict[str, object], action_dim: int) -> tuple[bool, ...]:
    mask = tuple(bool(value) for value in info.get("action_mask", ()))
    if len(mask) != action_dim:
        raise RuntimeError(f"action mask length mismatch: {len(mask)} != {action_dim}")
    return mask


def _evaluation_key(record: dict[str, Any]) -> tuple[float, float]:
    return (
        float(record.get("win_rate", 0.0) or 0.0),
        float(record.get("mean_score", 0.0) or 0.0),
    )


class DoubleDQNTrainerController:
    """Own one Double DQN training run and expose pause/resume/stop/checkpoint controls."""

    def __init__(
        self,
        config: DoubleDQNConfig,
        *,
        dll_path: str | None = None,
        seed_types: tuple[int, ...] | None = None,
        run_dir: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        on_event: EventCallback | None = None,
        eval_every_episodes: int = _DEFAULT_EVAL_EVERY_EPISODES,
        eval_seed_start: int = _DEFAULT_EVAL_SEED_START,
        eval_seed_count: int = _DEFAULT_EVAL_SEED_COUNT,
        eval_max_episode_steps: int | None = _DEFAULT_EVAL_MAX_EPISODE_STEPS,
    ) -> None:
        if eval_every_episodes < 0:
            raise ValueError("eval_every_episodes must be non-negative")
        if eval_seed_start < 0:
            raise ValueError("eval_seed_start must be non-negative")
        if eval_seed_count <= 0:
            raise ValueError("eval_seed_count must be positive")
        if eval_max_episode_steps is not None and eval_max_episode_steps <= 0:
            raise ValueError("eval_max_episode_steps must be positive")

        if config.env.seed_selection_mode == "custom" and seed_types is not None:
            config = replace(config, env=replace(config.env, seed_priority=seed_types))
        self.config = config
        self.dll_path = dll_path
        self.seed_types = tuple(seed_types) if seed_types else None
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        level_id = int(config.env.level_id)
        if getattr(config.env, "level_pool", None):
            level_tag = f"mixed-{len(config.env.level_pool)}l"
        elif 1 <= level_id <= 50:
            level_tag = f"{(level_id - 1) // 10 + 1}-{(level_id - 1) % 10 + 1}"
        else:
            level_tag = str(level_id)
        self.run_dir = (
            Path(run_dir)
            if run_dir is not None
            else Path("runs") / f"double-dqn-{level_tag}-{stamp}"
        )
        self.on_event = on_event
        self.eval_every_episodes = int(eval_every_episodes)
        self.eval_seed_start = int(eval_seed_start)
        self.eval_seed_count = int(eval_seed_count)
        if not hasattr(self, "_legacy_evaluation_protocol"):
            self._legacy_evaluation_protocol = (eval_seed_count != _DEFAULT_EVAL_SEED_COUNT
                                                or eval_seed_start != _DEFAULT_EVAL_SEED_START)
        self.eval_max_episode_steps = (None if eval_max_episode_steps is None
                                       else int(eval_max_episode_steps))
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.stop_event = threading.Event()
        self._agent_lock = threading.RLock()
        self._replay_lock = threading.RLock()
        self._agent: DoubleDQNAgent | None = None
        self._replay: ReplayBuffer | None = None
        self._summary = DoubleDQNSummary(config.total_steps)
        self._state = "READY"
        self._latest_checkpoint: Path | None = None
        self._best_eval_key: tuple[float, float] = (-math.inf, -math.inf)
        self._episode_rng = np.random.default_rng(config.seed)
        from train.adaptive_curriculum import AdaptiveCurriculum
        self._curriculum = AdaptiveCurriculum(config)
        self._checkpoint_reward_settings: dict[str, Any] | None = None
        self._checkpoint_gamma: float | None = None
        self._checkpoint_gamma_reference_ticks: int | None = None
        self._env: Any = None
        self._control_lock = threading.RLock()
        self._pause_acknowledged = False
        self._reset_requested = False
        self._runtime_pause_hook = None
        self._runtime_reset_hook = None
        self._async_learner = None
        self._user_pause_token = {"owned": False}
        from .visualization import TrainingVisualization
        self._visualization = TrainingVisualization(self)
        self._checkpoint_writer = None
        self._checkpoint_replay_file = None

    @property
    def state(self) -> str:
        return self._state

    @property
    def summary(self) -> DoubleDQNSummary:
        return self._summary

    @property
    def latest_checkpoint(self) -> Path | None:
        return self._latest_checkpoint

    def _emit(self, event: str, **payload: Any) -> None:
        if event in ("worker_recovery", "evaluation_retry", "evaluation_error", "error"):
            try:
                self.run_dir.mkdir(parents=True, exist_ok=True)
                with (self.run_dir / "runtime_events.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(dict(timestamp=datetime.now().astimezone().isoformat(),
                                                 event=event, **payload), ensure_ascii=False) + "\n")
            except OSError:
                logger.exception("Could not save training runtime event %s", event)
        if self.on_event is not None:
            payload.setdefault("summary", self._summary.snapshot())
            self.on_event(event, payload)

    def pause(self) -> None:
        with self._control_lock:
            if self._state == "RUNNING":
                self.pause_event.clear()
                self._state = "PAUSING"
                self._emit("state", state=self._state, acknowledged=False)

    def resume(self) -> None:
        with self._control_lock:
            if self._state == "PAUSED":
                self._state = "RESUMING"
                self._emit("state", state=self._state, acknowledged=False)
                self.pause_event.set()

    def reset_game(self) -> None:
        """Reset at a confirmed pause, preserving the learner and Replay."""
        with self._control_lock:
            if self._state != "PAUSED":
                raise RuntimeError("pause and wait for acknowledgement before Reset Game")
            if getattr(self, "_evaluation_active", False):
                raise RuntimeError("Reset Game is unavailable during checkpoint evaluation")
            self._reset_requested = True
            self._state = "RESETTING"
            self._emit("state", state=self._state, acknowledged=False)

    def set_visualization_enabled(self, enabled: bool) -> None:
        with self._agent_lock:
            self._visualize_q = bool(enabled)
            self.config = replace(self.config, runtime=replace(self.config.runtime, visualize_q=bool(enabled)))
            if self._agent is not None and hasattr(self._agent, "config"):
                self._agent.config = self.config
            if (self.run_dir / "config.json").is_file():
                self._persist_run_config()
        if not enabled:
            self._visualization.clear()

    def stop(self) -> None:
        if self._state in ("RUNNING", "PAUSING", "PAUSED", "RESUMING", "RESETTING"):
            self._visualization.clear()
            self._state = "STOPPING"
            self.stop_event.set()
            self.pause_event.set()
            self._emit("state", state=self._state)

    def set_simulation_mode(self, mode: str) -> None:
        """动态更新训练环境的仿真模式。"""
        from train.runtime import uses_parallel_runtime
        if (self._state in ('RUNNING','PAUSING','PAUSED','RESUMING','RESETTING','STOPPING')
                and uses_parallel_runtime(self.config.runtime)
                and mode != self.config.env.simulation_mode):
            raise RuntimeError('stop the process sampler before changing its simulation mode')
        if self._env is not None and hasattr(self._env, "set_simulation_mode"):
            self._env.set_simulation_mode(mode)
            current_mode = str(self._env.simulation_mode)
            with self._agent_lock:
                self.config = replace(self.config, env=replace(self.config.env, simulation_mode=current_mode))
                if self._agent is not None:
                    self._agent.config = self.config
                if (self.run_dir / "config.json").is_file():
                    self._persist_run_config()
            self._emit("simulation_mode", mode=current_mode)

    def set_exploration_settings(
        self,
        *,
        mode: str | None = None,
        ratio: Sequence[float] | None = None,
        type_weights: Sequence[float] | None = None,
    ) -> None:
        """动态热更新智能体的探索模式、混合比例与动作家族权重。"""
        updates = {}
        if mode is not None:
            updates["exploration_mode"] = str(mode)
        if ratio is not None:
            updates["exploration_ratio"] = tuple(float(x) for x in ratio)
        if type_weights is not None:
            updates["exploration_type_weights"] = tuple(float(x) for x in type_weights)
        if updates:
            with self._agent_lock:
                self.config = replace(self.config, **updates)
                if self._agent is not None:
                    self._agent.config = self.config
                if (self.run_dir / "config.json").is_file():
                    self._persist_run_config()
            self._emit("exploration_settings_updated", **updates)

    def set_replay_settings(
        self,
        *,
        scheme: str | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        eps: float | None = None,
    ) -> None:
        """动态热更新回放机制（uniform 或 per）及 PER 超参数。"""
        updates = {}
        if scheme is not None:
            updates["replay_scheme"] = str(scheme)
        if alpha is not None:
            updates["per_alpha"] = float(alpha)
        if beta is not None:
            updates["per_beta_start"] = float(beta)
        if eps is not None:
            updates["per_eps"] = float(eps)

        if updates:
            with self._replay_lock:
                validated_config = replace(self.config, **updates)
                if self._replay is not None:
                    self._replay.set_scheme(
                        validated_config.replay_scheme,
                        alpha=updates.get("per_alpha", self.config.per_alpha),
                        beta=updates.get("per_beta_start", self.config.per_beta_start),
                        eps=updates.get("per_eps", self.config.per_eps),
                    )
                with self._agent_lock:
                    self.config = validated_config
                    if self._agent is not None:
                        self._agent.config = self.config
                    if (self.run_dir / "config.json").is_file():
                        self._persist_run_config()
            self._emit("replay_settings_updated", **updates)

    def set_replay_scheme(
        self,
        scheme: str,
        *,
        alpha: float | None = None,
        beta: float | None = None,
        eps: float | None = None,
    ) -> None:
        self.set_replay_settings(scheme=scheme, alpha=alpha, beta=beta, eps=eps)

    def _persist_run_config(self) -> None:
        """Keep the run-folder baseline aligned with the effective agent config."""
        with self._agent_lock:
            payload = asdict(self.config)
            payload["seed_types"] = list(self.seed_types) if self.seed_types else None
            if self._agent is not None:
                payload["hidden_sizes"] = self._agent.hidden_sizes
                payload["credit_assignment"] = self._agent.credit_assignment
            destination = self.run_dir / "config.json"
            temporary = destination.with_suffix(".json.tmp")
            try:
                temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                temporary.replace(destination)
            finally:
                if temporary.exists():
                    temporary.unlink()

    def _wait_if_paused(self) -> bool:
        if self._checkpoint_writer is not None:
            self._checkpoint_writer.poll()
        while True:
            if self._async_learner is not None:
                self._async_learner.check()
            if self.stop_event.is_set():
                return False
            try:
                if not self.pause_event.is_set():
                    if not self._pause_acknowledged:
                        self._visualization.clear()
                        acknowledgements = self._apply_runtime_pause(True)
                        self._pause_acknowledged = True
                        with self._control_lock:
                            if not self.stop_event.is_set():
                                self._state = "PAUSED"
                                self._emit("state", state=self._state, acknowledged=True,
                                           control_ack=acknowledgements)
                    if self._reset_requested and not self.stop_event.is_set():
                        if self._runtime_reset_hook is None:
                            raise RuntimeError("Reset Game is unavailable in this runtime")
                        self._visualization.clear()
                        acknowledgements = self._runtime_reset_hook()
                        self._reset_requested = False
                        with self._control_lock:
                            if not self.stop_event.is_set():
                                self._state = "PAUSED"
                                self._emit("state", state=self._state, acknowledged=True,
                                           operation="reset", control_ack=acknowledgements)
                    if self._checkpoint_writer is not None:
                        self._checkpoint_writer.poll()
                    self.pause_event.wait(0.05)
                    continue
                if self._pause_acknowledged:
                    acknowledgements = self._apply_runtime_pause(False)
                    self._pause_acknowledged = False
                    with self._control_lock:
                        if not self.stop_event.is_set():
                            self._state = "RUNNING"
                            self._emit("state", state=self._state, acknowledged=True,
                                       control_ack=acknowledgements)
                return not self.stop_event.is_set()
            except InterruptedError:
                if self.stop_event.is_set():
                    return False
                raise
            except Exception as exc:
                self._state = "ERROR"
                self._emit("state", state=self._state, acknowledged=False, error=str(exc))
                raise

    def _apply_runtime_pause(self, paused):
        if self._runtime_pause_hook is not None:
            return self._runtime_pause_hook(paused)
        if self._env is not None:
            from train.sampler import set_environment_paused
            return [set_environment_paused(self._env, paused, self._user_pause_token)]
        raise RuntimeError("training runtime has not initialized its pause controls")

    def _replay_path(self) -> Path:
        return self.run_dir / "replay.npz"

    def save_checkpoint(
        self,
        path: str | Path | None = None,
        *,
        _internal: bool = False,
        force_replay: bool = False,
    ) -> Path:
        if self._state in ("RUNNING", "PAUSING", "PAUSED", "RESUMING", "RESETTING", "STOPPING", "ERROR") and not _internal:
            raise RuntimeError(
                "manual checkpoint save is disabled while the trainer is active; "
                "use autosave/milestone checkpoints or stop the run first"
            )
        self._flush_checkpoint_writer()
        agent = self._agent
        if agent is None:
            raise RuntimeError("trainer has not created an agent yet")
        destination = Path(path) if path else self.run_dir / "manual.pt"
        env_step = self._summary.env_steps
        episode = self._summary.episodes
        is_stop_or_terminal = self._state in ("STOPPED", "STOPPING", "FINISHED") or destination.name in ("stopped.pt", "final.pt")
        save_replay_forced = force_replay or is_stop_or_terminal
        if callable(getattr(agent, 'checkpoint_payload', None)):
            self._submit_checkpoint_snapshot(destination, force_replay=save_replay_forced)
            self._flush_checkpoint_writer()
            pair = getattr(self, "_last_saved_replay_path", None)
            if pair is not None and pair.is_file():
                legacy = self._replay_path()
                temporary = legacy.with_suffix('.npz.tmp')
                try:
                    shutil.copy2(pair, temporary)
                    temporary.replace(legacy)
                finally: temporary.unlink(missing_ok=True)
            return destination
        with self._agent_lock:
            saved = agent.save(
                destination,
                env_step=env_step,
                episode=episode,
                training_summary=self._summary.snapshot(),
                episode_rng_state=self._episode_rng.bit_generator.state,
                seed_types=self.seed_types,
            )
        replay = self._replay
        if replay is not None:
            with self._replay_lock:
                replay_path = replay.save(
                    self._replay_path(),
                    env_step=env_step,
                    episode=episode,
                    gamma=self.config.gamma,
                    gamma_reference_ticks=self.config.gamma_reference_ticks,
                    reward_settings=effective_reward_settings(self.config),
                )
            self._emit(
                "replay_checkpoint",
                path=str(replay_path),
                env_step=env_step,
                replay_size=len(replay),
            )
        self._latest_checkpoint = saved
        self._summary.checkpoint = str(saved)
        self._emit("checkpoint", path=str(saved))
        return saved

    def _flush_checkpoint_writer(self, *, close=False):
        if self._checkpoint_writer is not None:
            if close:
                writer, self._checkpoint_writer = self._checkpoint_writer, None
                writer.close()
            else: self._checkpoint_writer.flush()

    def _save_scheduled_checkpoint(self, path, *, sync_latest=False):
        if not self.config.runtime.asynchronous_checkpoints:
            saved = self.save_checkpoint(path, _internal=True)
            if sync_latest: self._sync_latest_checkpoint(saved)
            return saved
        return self._submit_checkpoint_snapshot(path, sync_latest=sync_latest)

    def _submit_checkpoint_snapshot(self, path, *, sync_latest=False, force_replay=False):
        from train.checkpoint_io import CheckpointWriter, freeze_checkpoint, freeze_replay
        if self._checkpoint_writer is None: self._checkpoint_writer = CheckpointWriter()
        self._checkpoint_writer.flush()
        step, episode = self._summary.env_steps, self._summary.episodes
        is_terminal = self._state in ("STOPPED", "STOPPING", "FINISHED") or Path(path).name in ("stopped.pt", "final.pt", "manual.pt")
        should_save_replay = self._replay is not None and (
            force_replay or is_terminal or not (
                Path(path).name in ('latest.pt', 'latest-model.pt') and getattr(self, '_model_only_autosave', False)
            )
        )
        with self._replay_lock, self._agent_lock:
            model = freeze_checkpoint(self._agent.checkpoint_payload(env_step=step, episode=episode,
                training_summary=self._summary.snapshot(), episode_rng_state=self._episode_rng.bit_generator.state,
                seed_types=self.seed_types))
            replay = self._replay.checkpoint_payload(env_step=step, episode=episode,
                gamma=self.config.gamma, gamma_reference_ticks=self.config.gamma_reference_ticks,
                reward_settings=effective_reward_settings(self.config),
                copy=False) if should_save_replay else None
            if replay is not None and self._state not in ("STOPPED", "FINISHED"):
                replay = freeze_replay(replay, directory=self.run_dir)
        def completed(destination, replay_file):
            self._last_saved_replay_path = replay_file if replay is not None else None
            self._latest_checkpoint = destination
            self._summary.checkpoint = str(destination)
            self._emit("checkpoint", path=str(destination), env_step=step,
                       kind="full" if replay is not None else "model_only",
                       strict_resume_supported=replay is not None, exact_game_resume=False)
            if replay is not None: self._emit("replay_checkpoint", path=str(replay_file), env_step=step)
        self._checkpoint_writer.submit(path, model, replay, sync_latest=sync_latest, on_complete=completed)
        return Path(path)

    def _sync_latest_checkpoint(self, source: Path) -> Path:
        destination = self.run_dir / "latest.pt"
        if source.resolve() != destination.resolve():
            temporary = destination.with_name(destination.name + ".tmp")
            try:
                shutil.copy2(source, temporary)
                temporary.replace(destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
        self._emit("checkpoint", path=str(destination), kind="latest")
        return destination

    def _restore_metrics_history(self, metrics_path: Path, start_episode: int) -> None:
        """Restore cumulative UI statistics for a same-directory resumed run."""
        if start_episode <= 0 or not metrics_path.is_file():
            return
        records: dict[int, dict[str, Any]] = {}
        try:
            with metrics_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        continue
                    episode = int(record.get("episode", 0) or 0)
                    if 0 < episode <= start_episode:
                        records[episode] = record
        except Exception:
            return
        if not records:
            return

        self._summary.wins = 0
        self._summary.losses = 0
        self._summary.invalid_actions = 0
        self._summary.mowers_used = 0
        self._summary.raw_episode_rewards.clear()
        self._summary.training_episode_rewards.clear()
        self._summary.episode_lengths.clear()
        for episode in sorted(records):
            record = records[episode]
            result = str(record.get("result", "UNKNOWN"))
            if result == "WON":
                self._summary.wins += 1
            elif result == "LOST":
                self._summary.losses += 1
            self._summary.invalid_actions += max(0, int(record.get("invalid_actions", 0) or 0))
            self._summary.mowers_used += max(0, int(record.get("mowers_used", 0) or 0))
            self._summary.raw_episode_rewards.append(float(record.get("raw_episode_reward", 0.0) or 0.0))
            self._summary.training_episode_rewards.append(
                float(record.get("training_episode_reward", 0.0) or 0.0)
            )
            self._summary.episode_lengths.append(max(0, int(record.get("episode_steps", 0) or 0)))
            loss = record.get("loss")
            if loss is not None:
                self._summary.last_loss = float(loss)

        last_record = records[max(records)]
        tick_samples = max(0, int(last_record.get("tick_samples", 0) or 0))
        tick_mean = float(last_record.get("mean_ticks_advanced", 0.0) or 0.0)
        tick_std = float(last_record.get("std_ticks_advanced", 0.0) or 0.0)
        self._summary.tick_samples = tick_samples
        self._summary.ticks_advanced_sum = int(round(tick_mean * tick_samples))
        self._summary.ticks_advanced_sq_sum = (
            (tick_std * tick_std + tick_mean * tick_mean) * tick_samples
        )
        self._summary.long_tick_jumps = max(
            0, int(last_record.get("long_tick_jumps", 0) or 0)
        )

    def _restore_curriculum(self, payload, start_episode):
        from train.adaptive_curriculum import AdaptiveCurriculum
        self._curriculum = AdaptiveCurriculum(self.config)
        restored = self._curriculum.restore(payload.get("curriculum_state")) if start_episode else False
        metrics = self.run_dir / "episodes.jsonl"
        if not restored and start_episode and metrics.is_file():
            records = {}
            with metrics.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            continue
                        if 0 < int(record.get("episode", 0)) <= start_episode:
                            records[int(record["episode"])] = record
                    except (ValueError, TypeError):
                        continue
            for episode in sorted(records):
                self._curriculum.observe(records[episode])
        self._agent.curriculum_state = self._curriculum.snapshot()
        self._emit("level_sampling", enabled=self._curriculum.enabled, levels=self._curriculum.rows())

    def _record_curriculum_episode(self, record):
        updated = self._curriculum.observe(record)
        self._agent.curriculum_state = self._curriculum.snapshot()
        if updated:
            self._emit("level_sampling", enabled=self._curriculum.enabled, levels=self._curriculum.rows())

    def _evaluation_protocol_matches(self, record: dict[str, Any]) -> bool:
        from .evaluation_plan import protocol_matches
        return protocol_matches(self, record, self.eval_seed_count)

    def _restore_evaluation_history(self, evaluations_path: Path, start_step: int) -> None:
        if not evaluations_path.is_file():
            return
        best_record: dict[str, Any] | None = None
        runs = 0
        try:
            with evaluations_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.strip()
                    if not text:
                        continue
                    record = json.loads(text)
                    if not isinstance(record, dict):
                        continue
                    if not bool(record.get("complete", False)):
                        continue
                    if int(record.get("env_step", 0) or 0) > start_step:
                        continue
                    if not self._evaluation_protocol_matches(record):
                        continue
                    runs += 1
                    if best_record is None or _evaluation_key(record) > _evaluation_key(best_record):
                        best_record = record
        except Exception:
            return
        self._summary.eval_runs = runs
        if best_record is None:
            return
        self._best_eval_key = _evaluation_key(best_record)
        self._summary.best_eval_win_rate = self._best_eval_key[0]
        self._summary.best_eval_mean_score = self._best_eval_key[1]
        best_path = self.run_dir / "best.pt"
        if best_path.is_file():
            self._summary.best_checkpoint = str(best_path)

    def _build_agent(
        self,
        action_dim: int,
    ) -> tuple[DoubleDQNAgent, int, int, dict[str, Any]]:
        if self.checkpoint_path is None:
            return DoubleDQNAgent(action_dim, self.config), 0, 0, {}

        agent, checkpoint_config, payload = DoubleDQNAgent.from_checkpoint(
            self.checkpoint_path,
            device=self.config.device,
            strict_resume=getattr(self, "checkpoint_load_mode", "resume") == "strict_resume",
        )
        validate_resume_config(self.config, checkpoint_config)
        if "seed_types" in payload:
            saved_cards = tuple(payload["seed_types"]) if payload["seed_types"] else None
            if self.seed_types != saved_cards:
                raise ValueError("接续训练不能修改卡组；请使用权重微调模式。")
        if agent.action_dim != action_dim:
            raise RuntimeError(
                f"checkpoint action dimension {agent.action_dim} does not match "
                f"environment {action_dim}"
            )
        agent.set_reward_config(self.config)
        for group in agent.optimizer.param_groups:
            group["lr"] = self.config.learning_rate
        start_step = int(payload.get("env_step", 0))
        start_episode = int(payload.get("episode", 0))
        from .checkpoint_contract import inspect_checkpoint
        self._checkpoint_kind = inspect_checkpoint(payload)
        self._emit("checkpoint_restore_contract", path=str(self.checkpoint_path),
                   checkpoint_kind=self._checkpoint_kind,
                   load_mode=getattr(self, "checkpoint_load_mode", "resume"),
                   exact_game_resume=False)
        if "observation_migration" in payload:
            self._emit("observation_migration", **payload["observation_migration"])
        self._checkpoint_observation_migration = payload.get("observation_migration")
        self._checkpoint_reward_settings = effective_reward_settings(checkpoint_config)
        self._checkpoint_gamma = checkpoint_config.gamma
        self._checkpoint_gamma_reference_ticks = checkpoint_config.gamma_reference_ticks
        replay_file = payload.get("replay_file")
        if replay_file:
            if Path(replay_file).name != replay_file: raise ValueError("invalid checkpoint replay_file")
            self._checkpoint_replay_file = self.checkpoint_path.parent / replay_file
        if "episode_rng_state" in payload:
            self._episode_rng.bit_generator.state = payload["episode_rng_state"]
        elif start_step > 0:
            self._episode_rng = np.random.default_rng(np.random.SeedSequence([
                checkpoint_config.seed, start_step, start_episode, 1,
            ]))
        self._latest_checkpoint = self.checkpoint_path
        self._summary.checkpoint = str(self.checkpoint_path)
        return agent, start_step, start_episode, payload

    def _new_replay(self, action_dim: int) -> ReplayBuffer:
        return ReplayBuffer(
            self.config.replay_capacity,
            action_dim,
            seed=self.config.seed + 1,
            credit_assignment=self.config.credit_assignment,
            scheme=self.config.replay_scheme,
            alpha=self.config.per_alpha,
            beta=self.config.per_beta_start,
            eps=self.config.per_eps,
            reward_context=self.config.reward_scheme == "transition_dense",
            boss_projectiles=self.config.boss_projectiles,
        )

    def _restore_replay(
        self,
        *,
        action_dim: int,
        start_step: int,
        start_episode: int,
    ) -> ReplayBuffer:
        if self.checkpoint_path is None:
            return self._new_replay(action_dim)

        from .checkpoint_contract import (CHECKPOINT_LOAD_STRICT, CHECKPOINT_LOAD_MODEL,
                                         CHECKPOINT_LOAD_WEIGHTS_ONLY, CHECKPOINT_LOAD_WEIGHTS_REPLAY)
        mode = getattr(self, "checkpoint_load_mode", "resume")
        require_pair = mode in (CHECKPOINT_LOAD_STRICT, CHECKPOINT_LOAD_WEIGHTS_REPLAY)
        restored = None

        def skipped(reason, path=None):
            nonlocal restored
            restored = None  # Release incompatible data before allocating an empty Replay.
            self._summary.replay_restored = False
            self._emit("replay_restore_skipped", path=str(path) if path else None, reason=reason,
                       load_mode=mode, warmup_restarted=not require_pair)
            if require_pair:
                raise ValueError(mode + " requires a matching full Replay: " + reason)
            return self._new_replay(action_dim)

        if mode == CHECKPOINT_LOAD_MODEL:
            return skipped("model_resume explicitly starts Replay empty while retaining model/optimizer/progress")
        if mode == CHECKPOINT_LOAD_WEIGHTS_ONLY:
            return skipped("weights_only starts Replay empty while retaining only network weights")
        if getattr(self, "_checkpoint_observation_migration", None):
            return skipped("Boss observation/reward semantics changed; retain model/optimizer/progress and rebuild Replay")
        if require_pair and getattr(self, "_checkpoint_kind", "legacy_unknown") == "model_only":
            return skipped("checkpoint is model-only and contains no paired Replay")

        # Paired modes must use the declared snapshot, even if another valid
        # Replay with identical counters exists beside it. Only ordinary resume
        # may discover a replacement sidecar.
        replay_path = None
        if require_pair:
            replay_path = self._checkpoint_replay_file
        elif self._checkpoint_replay_file and self._checkpoint_replay_file.is_file():
            replay_path = self._checkpoint_replay_file
        else:
            search_dirs = []
            if self.checkpoint_path and self.checkpoint_path.parent.is_dir():
                search_dirs.append(self.checkpoint_path.parent)
            if self.run_dir and self.run_dir.is_dir() and self.run_dir not in search_dirs:
                search_dirs.append(self.run_dir)

            candidates: list[tuple[int, int, Path]] = []
            for d in search_dirs:
                for f in d.glob("*.npz"):
                    if not f.is_file():
                        continue
                    mtime = f.stat().st_mtime_ns
                    step_val = -1
                    stem = f.stem
                    if "-replay-" in stem:
                        try:
                            parts = stem.split("-replay-")
                            step_val = int(parts[1].split("-")[0])
                        except Exception:
                            pass
                    candidates.append((step_val, mtime, f))
            if candidates:
                # 优先匹配步数最高（最新进度），若无步数信息则按修改时间最新排序
                candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
                replay_path = candidates[0][2]
            elif self._replay_path().is_file():
                replay_path = self._replay_path()

        if replay_path is None or not replay_path.is_file():
            return skipped("Replay sidecar is missing", replay_path)

        try:
            restored, metadata = ReplayBuffer.load(
                replay_path,
                capacity=self.config.replay_capacity,
                action_dim=action_dim,
                seed=self.config.seed + 1,
                credit_assignment=self.config.credit_assignment,
                strict_resume=mode == CHECKPOINT_LOAD_STRICT,
            )
            from .reward_observation import observation_schema
            expected_schema = observation_schema(self.config.reward_scheme == "transition_dense", self.config.boss_projectiles)
            if restored.observation_schema != expected_schema:
                raise ValueError("Replay observation schema does not match this network's input contract")
            if self.config.n_step > 1 and not restored.supports_n_step:
                raise ValueError("legacy replay lacks episode boundaries for n-step returns; rebuild replay")
        except Exception as exc:
            reason = str(exc)
            exc.__traceback__ = None  # A failed loader's frame may own large arrays.
            return skipped(reason, replay_path)

        replay_step = int(metadata.get("env_step", -1))
        replay_episode = int(metadata.get("episode", -1))
        if replay_step != start_step or replay_episode != start_episode:
            if require_pair:
                return skipped(
                        f"snapshot step/episode {replay_step}/{replay_episode} does not match "
                        f"checkpoint {start_step}/{start_episode}", replay_path)
            # 在普通 resume 模式下，容忍步数与回合数的偏差，保留并恢复最新的经验池，避免冷启动导致 TD Loss 暴增
            self._emit("replay_step_mismatch_tolerated",
                       replay_step=replay_step, start_step=start_step,
                       replay_episode=replay_episode, start_episode=start_episode,
                       path=str(replay_path))
        current_reward = effective_reward_settings(self.config)
        if mode == CHECKPOINT_LOAD_WEIGHTS_REPLAY and metadata.get("credit_assignment") != self.config.credit_assignment:
            return skipped("Replay credit assignment does not match the new training", replay_path)
        saved_reward = metadata.get("reward_settings", self._checkpoint_reward_settings)
        if saved_reward is not None and saved_reward != current_reward:
            return skipped("reward settings changed; rebuild replay while retaining model/optimizer state", replay_path)
        if (restored.scheme != self.config.replay_scheme or restored.alpha != self.config.per_alpha
                or restored.beta != self.config.per_beta_start or restored.eps != self.config.per_eps):
            restored.set_scheme(
                self.config.replay_scheme,
                alpha=self.config.per_alpha,
                beta=self.config.per_beta_start,
                eps=self.config.per_eps,
            )
        restored.reward_settings = current_reward
        self._summary.replay_restored = True
        self._emit(
            "replay_restored",
            path=str(replay_path),
            env_step=replay_step,
            episode=replay_episode,
            replay_size=len(restored),
        )
        return restored

    def _is_evaluation_due(self) -> bool:
        if getattr(self.config, "evaluation_mode", None) in ("off", "none"):
            return False
        return (
            self.eval_every_episodes > 0
            and self._summary.episodes > 0
            and self._summary.episodes % self.eval_every_episodes == 0
        )

    def _evaluate_current_agent(
        self,
        env: PvZEnv,
        action_dim: int,
        *,
        env_step: int,
    ) -> dict[str, Any]:
        agent = self._agent
        if agent is None:
            raise RuntimeError("trainer has not created an agent yet")
        from .evaluation_plan import evaluation_plan, evaluation_mode, protocol_fields, record_level_result
        tasks = evaluation_plan(self, self.eval_seed_count)
        requested = len(tasks)
        evaluation_seed_start = tasks[0][0]
        level_results = {}
        wins = losses = unknown = total_steps = total_ticks = completed = 0
        total_score = 0.0
        total_training_score = 0.0
        zero_mower_wins = win_mowers = all_mowers = 0
        stopped = False
        error: str | None = None
        self._emit(
            "evaluation_start",
            env_step=env_step,
            episode=self._summary.episodes,
            seed_start=evaluation_seed_start,
            requested_episodes=requested,
            evaluation_mode=evaluation_mode(self),
        )

        for seed, level_pair in tasks:
            if not self._wait_if_paused():
                stopped = True
                break
            try:
                env.level_id, env.playthrough = level_pair
                observation, info = env.reset(seed=seed)
                actual_pair = (info.get('level_id', level_pair[0]), info.get('playthrough', level_pair[1]))
                if actual_pair != level_pair:
                    raise RuntimeError(f"evaluation level mismatch: {actual_pair} != {level_pair}")
                reward_tracker = None
                if self.config.reward_scheme == "transition_dense":
                    from .transition_reward import TransitionDenseReward
                    reward_tracker = TransitionDenseReward(info, shaping_limit_mode=self.config.shaping_limit_mode)
                    observation = reward_tracker.observe(observation)
                if self.config.reward_scheme == "staged":
                    from .staged_reward import StagedReward
                    reward_tracker = StagedReward(info)
                training_score = 0.0
                observation = normalize_observation(observation, boss_projectiles=self.config.boss_projectiles)
                mask = _require_mask(info, action_dim)
                result = "UNKNOWN"
                score = 0.0
                steps = 0
                terminated = truncated = False
                first_mower_seen = False
                first_mower_tick = None
                episode_ticks = 0
                episode_mowers = 0

                while self.eval_max_episode_steps is None or steps < self.eval_max_episode_steps:
                    if not self._wait_if_paused():
                        stopped = True
                        break
                    action = agent.select_action(
                        observation,
                        mask,
                        env_step=env_step,
                        explore=False,
                    )
                    next_observation, reward, terminated, truncated, info = env.step(action)
                    if reward_tracker is not None:
                        at_step_limit = (self.eval_max_episode_steps is not None
                                         and steps + 1 >= self.eval_max_episode_steps)
                        training_score += reward_tracker.step(reward, info, terminated=terminated,
                                                       truncated=truncated or at_step_limit)
                    episode_ticks += int(info.get("ticks_advanced", self.config.env.ticks_per_step))
                    episode_mowers += mower_event_count(info.get("events"))
                    if not first_mower_seen and mower_event_count(info.get("events")) > 0:
                        first_mower_seen = True
                        first_mower_tick = episode_ticks
                    if self.config.reward_scheme == "transition_dense":
                        next_observation = reward_tracker.observe(next_observation)
                    observation = normalize_observation(next_observation, boss_projectiles=self.config.boss_projectiles)
                    mask = _require_mask(info, action_dim)
                    result = str(info.get("result", "UNKNOWN"))
                    score += float(reward)
                    steps += 1
                    if terminated or truncated:
                        break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                self._emit(
                    "evaluation_error",
                    env_step=env_step,
                    episode=self._summary.episodes,
                    seed=seed,
                    error=error,
                )
                break

            if stopped:
                break
            completed += 1
            record_level_result(level_results, level_pair, result)
            wins += int(result == "WON")
            losses += int(result == "LOST")
            unknown += int(result not in ("WON", "LOST"))
            zero_mower_wins += int(result == "WON" and episode_mowers == 0)
            win_mowers += episode_mowers if result == "WON" else 0
            all_mowers += episode_mowers
            total_score += score
            total_training_score += training_score
            total_steps += steps
            total_ticks += episode_ticks
            self._emit(
                "evaluation_episode",
                env_step=env_step,
                episode=self._summary.episodes,
                seed=seed,
                level_id=level_pair[0], playthrough=level_pair[1],
                result=result,
                score=score,
                steps=steps,
                episode_ticks=episode_ticks,
                first_mower_tick=first_mower_tick,
                mowers_used=episode_mowers,
                completed=completed,
                requested_episodes=requested,
                **({"training_score": training_score,
                    "reward_breakdown": reward_tracker.last_breakdown} if reward_tracker is not None else {}),
            )

        record = {
            "env_step": int(env_step),
            "episode": int(self._summary.episodes),
            "seed_start": evaluation_seed_start,
            "requested_episodes": requested,
            "max_episode_steps": self.eval_max_episode_steps,
            "max_episode_ticks": self.config.env.max_episode_ticks,
            "episodes": completed,
            "wins": wins,
            "losses": losses,
            "unknown": unknown,
            "win_rate": wins / completed if completed else 0.0,
            "zero_mower_wins": zero_mower_wins,
            "zero_mower_win_rate": zero_mower_wins / completed if completed else 0.0,
            "mean_win_mowers": win_mowers / wins if wins else None,
            "mean_mowers": all_mowers / completed if completed else 0.0,
            "mean_score": total_score / completed if completed else 0.0,
            "mean_steps": total_steps / completed if completed else 0.0,
            "mean_episode_ticks": total_ticks / completed if completed else 0.0,
            "complete": completed == requested and not stopped and error is None,
            "stopped": stopped,
            "error": error,
        }
        if self.config.reward_scheme in ("staged", "transition_dense"):
            record["mean_training_score"] = total_training_score / completed if completed else 0.0
        record.update(protocol_fields(self, requested), levels=list(level_results.values()))
        self._emit("evaluation_end", **record)
        return record

    def _record_evaluation(self, record: dict[str, Any], evaluations_path: Path) -> None:
        with evaluations_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if not bool(record.get("complete", False)):
            return

        self._summary.eval_runs += 1
        score_key = _evaluation_key(record)
        if score_key <= self._best_eval_key:
            return

        self._best_eval_key = score_key
        self._summary.best_eval_win_rate = score_key[0]
        self._summary.best_eval_mean_score = score_key[1]
        best_path = self.run_dir / "best.pt"
        agent = self._agent
        if agent is None:
            raise RuntimeError("trainer has not created an agent yet")
        self._summary.best_checkpoint = str(best_path)
        with self._agent_lock:
            agent.save(
                best_path,
                env_step=self._summary.env_steps,
                episode=self._summary.episodes,
                training_summary=self._summary.snapshot(),
                episode_rng_state=self._episode_rng.bit_generator.state,
                seed_types=self.seed_types,
            )
        best_metadata = dict(record)
        best_metadata["checkpoint"] = str(best_path)
        (self.run_dir / "best.json").write_text(
            json.dumps(best_metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self._emit(
            "best_checkpoint",
            path=str(best_path),
            win_rate=score_key[0],
            mean_score=score_key[1],
            env_step=self._summary.env_steps,
            episode=self._summary.episodes,
        )

    def run(self) -> DoubleDQNSummary:
        if self._state not in ("READY", "STOPPED", "FINISHED"):
            raise RuntimeError(f"trainer cannot start from state {self._state}")

        if self.config.training_architecture == "actor_learner" or uses_parallel_runtime(self.config.runtime):
            from .parallel_controller import run_parallel
            return run_parallel(self)
        require_runtime_implementation(self.config.env.simulation_mode, self.config.env.runtime)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = self.run_dir / "episodes.jsonl"
        evaluations_path = self.run_dir / "evaluations.jsonl"
        env = PvZEnv(
            PvZClient(
                dll_path=self.dll_path,
                automation_enabled=True,
                auto_dialogue=True,
                auto_collect=True,
                auto_select_seeds=True,
                auto_skip_tutorial=True,
                background_running=True,
                deterministic_pause=True,
                runtime_config=self.config.env.runtime,
            ),
            level_id=self.config.env.level_id,
            playthrough=self.config.env.playthrough,
            seed_types=self.seed_types or self.config.env.seed_priority,
            seed_selection_mode=self.config.env.seed_selection_mode,
            ticks_per_step=self.config.env.ticks_per_step,
            max_episode_ticks=self.config.env.max_episode_ticks,
            simulation_mode=self.config.env.simulation_mode,
            runtime_config=self.config.env.runtime,
            turbo_config=self.config.env.turbo,
            reward_stage_ticks=1000 if self.config.reward_scheme == "staged" else None,
        )
        self._env = env

        self._state = "RUNNING"
        self.stop_event.clear()
        self.pause_event.set()
        self._pause_acknowledged = False
        self._reset_requested = False
        self._user_pause_token = {"owned": False}
        self._visualization.clear()
        self._emit("state", state=self._state)

        timing = TimingRecorder()
        try:
            action_dim = int(env.action_space.n)
            self._agent, start_step, start_episode, checkpoint_payload = self._build_agent(
                action_dim
            )
            self._persist_run_config()
            self._replay = self._restore_replay(
                action_dim=action_dim,
                start_step=start_step,
                start_episode=start_episode,
            )
            replay = self._replay
            self._emit(
                "replay_ready", configured_scheme=self.config.replay_scheme,
                effective_scheme=replay.scheme, alpha=replay.alpha,
                beta=replay.beta, eps=replay.eps, restored=self._summary.replay_restored,
            )
            self._summary.env_steps = start_step
            self._summary.episodes = start_episode
            self._summary.replay_size = len(replay)
            self._summary.epsilon = self._agent.epsilon(start_step)
            self._restore_metrics_history(metrics_path, start_episode)
            self._restore_curriculum(checkpoint_payload, start_episode)
            self._restore_evaluation_history(evaluations_path, start_step)

            if self.checkpoint_path is not None:
                self._emit(
                    "checkpoint_loaded",
                    path=str(self.checkpoint_path),
                    env_step=start_step,
                    episode=start_episode,
                    checkpoint_summary=(
                        dict(checkpoint_payload.get("training_summary", {}))
                        if isinstance(checkpoint_payload.get("training_summary"), dict)
                        else {}
                    ),
                    replay_restored=self._summary.replay_restored,
                    replay_size=len(replay),
                )

            if start_step >= self.config.total_steps:
                self._state = "FINISHED"
                self._emit("finished")
                return self._summary

            if getattr(self.config.env, "level_pool", None):
                cur_lvl, cur_p = self._curriculum.sample(self._episode_rng)
                env.level_id = cur_lvl
                env.playthrough = cur_p
            state, info = timing.measure("reset", env.reset, seed=_episode_seed(self._episode_rng))
            from train.curriculum import episode_level_metadata
            episode_metadata = episode_level_metadata(info, self.config.env, env)
            self._emit("seed_selection", **info.get("seed_selection", {}))
            state = normalize_observation(state)
            mask = _require_mask(info, action_dim)
            credit = EpisodeEventCredit(replay, self.config, info)
            state = credit.observe(state)
            episode_steps = 0
            raw_episode_reward = 0.0
            train_episode_reward = 0.0
            episode_mowers = 0
            episode_invalid = 0
            last_result = "UNKNOWN"
            previous_replay_eligible = bool(info.get("replay_eligible", True))
            visual_episode = start_episode + 1

            def bind_visualization(reset_info):
                bound_info = dict(reset_info)
                pid = (getattr(env, "runtime_process_id", None)
                       or self.config.env.runtime.process_id)
                if not pid and self._visualization.enabled():
                    read_snapshot = getattr(getattr(env, "client", None), "read_snapshot", None)
                    if callable(read_snapshot):
                        try:
                            pid = read_snapshot().identity.process_id
                        except Exception as exc:
                            self._emit("visualization_unavailable", error=str(exc))
                if type(pid) is int:
                    bound_info["runtime_process_id"] = pid
                self._visualization.bind(bound_info, episode_id=visual_episode)

            bind_visualization(info)

            def reset_serial_training():
                nonlocal state, mask, credit, episode_steps, raw_episode_reward
                nonlocal train_episode_reward, episode_mowers, episode_invalid, last_result
                nonlocal previous_replay_eligible, visual_episode
                nonlocal episode_metadata
                with self._replay_lock:
                    replay.end_episode()
                self._emit("episode_aborted", reason="user_reset", episode_id=visual_episode,
                           episode_steps=episode_steps, env_step=self._summary.env_steps)
                self._user_pause_token.clear()
                state, reset_info = timing.measure("reset", env.reset, seed=_episode_seed(self._episode_rng))
                episode_metadata = episode_level_metadata(reset_info, self.config.env, env)
                state = normalize_observation(state)
                mask = _require_mask(reset_info, action_dim)
                credit = EpisodeEventCredit(replay, self.config, reset_info)
                state = credit.observe(state)
                previous_replay_eligible = bool(reset_info.get("replay_eligible", True))
                visual_episode += 1
                bind_visualization(reset_info)
                self._emit("seed_selection", **reset_info.get("seed_selection", {}))
                episode_steps = episode_mowers = episode_invalid = 0
                raw_episode_reward = train_episode_reward = 0.0
                last_result = "UNKNOWN"
                return self._apply_runtime_pause(True)

            self._runtime_reset_hook = reset_serial_training
            replay_warmup = max(
                self.config.batch_size,
                min(self.config.learning_starts, self.config.replay_capacity),
            )

            for env_step in range(start_step + 1, self.config.total_steps + 1):
                if not timing.measure("pause_wait", self._wait_if_paused):
                    self._summary.stopped = True
                    break

                step_started = perf_counter() if timing.enabled else 0.0
                if self._visualization.enabled() and not self._visualization.binding.get("process_id"):
                    bind_visualization(info)
                action = timing.measure("policy", self._agent.select_action,
                    state,
                    mask,
                    env_step=env_step - 1,
                    explore=True,
                    **({"visualize": True} if self._visualization.enabled() else {}),
                )
                visualizations = getattr(self._agent, "last_visualizations", ())
                if self._visualization.enabled() and visualizations:
                    self._visualization.publish(visualizations[0], sequence_id=episode_steps,
                                                policy_version=self._agent.update_count)
                next_state_raw, raw_reward, terminated, truncated, next_info = timing.measure("env_step", env.step, action)
                next_state = timing.measure("normalize", normalize_observation, next_state_raw)
                next_mask = _require_mask(next_info, action_dim)
                done = bool(terminated or truncated)
                events = next_info.get("events") or {}
                current_replay_eligible = bool(next_info.get("replay_eligible", True))
                transition_eligible = previous_replay_eligible and current_replay_eligible
                with self._replay_lock:
                    if transition_eligible:
                        train_reward = timing.measure("replay_add", credit.add,
                            replay, state, action, raw_reward, next_state,
                            terminated, truncated, next_mask, next_info,
                        )
                    else:
                        replay.end_episode()
                        train_reward = 0.0
                        credit = EpisodeEventCredit(replay, self.config, next_info)
                        next_state = credit.observe(next_state)
                if not transition_eligible:
                    self._visualization.clear()
                    self._emit("transition_discarded", env_step=env_step,
                               observation_quality=next_info.get("observation_quality"),
                               previous_observation_eligible=previous_replay_eligible,
                               error=next_info.get("observation_error"))
                previous_replay_eligible = current_replay_eligible

                action_result = next_info.get("action_result") or {}
                if isinstance(action_result, dict) and not bool(action_result.get("success", False)):
                    episode_invalid += 1
                    self._summary.invalid_actions += 1

                step_mowers = 0
                if isinstance(events, dict):
                    try:
                        step_mowers = max(0, int(events.get("mowers_used", 0) or 0))
                    except (TypeError, ValueError, OverflowError):
                        step_mowers = 0
                episode_mowers += step_mowers
                self._summary.mowers_used += step_mowers
                episode_steps += 1
                raw_episode_reward += float(raw_reward)
                train_episode_reward += train_reward
                last_result = str(next_info.get("result", "UNKNOWN"))
                ticks_requested = max(
                    1,
                    int(next_info.get("ticks_requested", self.config.env.ticks_per_step) or 1),
                )
                ticks_advanced = max(
                    0,
                    int(next_info.get("ticks_advanced", ticks_requested) or 0),
                )
                self._summary.tick_samples += 1
                self._summary.ticks_advanced_sum += ticks_advanced
                self._summary.ticks_advanced_sq_sum += float(ticks_advanced * ticks_advanced)
                if ticks_advanced > ticks_requested * 2:
                    self._summary.long_tick_jumps += 1

                if (transition_eligible and len(replay) >= replay_warmup and replay.can_sample(self.config.n_step)
                        and env_step % self.config.train_every == 0):
                    beta_frac = min(1.0, max(0.0, float(env_step) / max(1, self.config.per_beta_steps)))
                    current_beta = self.config.per_beta_start + beta_frac * (self.config.per_beta_end - self.config.per_beta_start)
                    for _ in range(self.config.gradient_steps):
                        with self._replay_lock:
                            batch = timing.measure("replay_sample", replay.sample, self.config.batch_size, n_step=self.config.n_step,
                                                  gamma=self.config.gamma,
                                                  gamma_reference_ticks=self.config.gamma_reference_ticks,
                                                  beta=current_beta)
                        with self._agent_lock:
                            self._summary.last_loss = timing.measure("learn", self._agent.learn, batch)
                        if replay.is_per and batch.indices is not None and self._agent.last_td_errors is not None:
                            with self._replay_lock:
                                timing.measure("per_priority_update", replay.update_priorities, batch.indices, self._agent.last_td_errors)

                if timing.enabled and (done or env_step % 1000 == 0):
                    timing.save(self.run_dir / "timing.json")
                    bridge_timing.save(self.run_dir / "bridge_timing.json")
                self._summary.env_steps = env_step
                self._summary.replay_size = len(replay)
                self._summary.epsilon = self._agent.epsilon(env_step)
                if (env_step % 10 == 0 or done or step_mowers
                        or (next_info.get("reward_breakdown") or {}).get("stage_settled")):
                    self._emit(
                        "step",
                        **episode_metadata,
                        env_step=env_step,
                        episode=self._summary.episodes + 1,
                        episode_step=episode_steps,
                        result=last_result,
                        game_clock=int(next_info.get("game_clock", 0)),
                        raw_reward=float(raw_reward),
                        training_reward=train_reward,
                        reward_scheme=self.config.reward_scheme,
                        mower_reward=((next_info.get("reward_breakdown") or {}).get("mower_reward",
                            effective_reward_settings(self.config)["coefficients"]["mower_used"] * step_mowers)),
                        reward_breakdown=next_info.get("reward_breakdown"),
                        mowers_used=step_mowers,
                        ticks_requested=ticks_requested,
                        ticks_advanced=ticks_advanced,
                    )

                if timing.enabled:
                    timing.add("training_step", (perf_counter() - step_started) * 1000)

                milestone_due = (
                    self.config.checkpoint_every
                    and env_step % self.config.checkpoint_every == 0
                )
                autosave_due = (
                    self.config.autosave_every
                    and env_step % self.config.autosave_every == 0
                )
                if milestone_due:
                    milestone = timing.measure("save_snapshot", self._save_scheduled_checkpoint,
                        self.run_dir / f"checkpoint-{env_step:09d}.pt",
                        sync_latest=bool(autosave_due),
                    )
                elif autosave_due:
                    timing.measure("save_snapshot", self._save_scheduled_checkpoint, self.run_dir / "latest.pt")

                state, mask = next_state, next_mask

                if done:
                    self._visualization.clear()
                    if last_result == "WON":
                        self._summary.wins += 1
                    elif last_result == "LOST":
                        self._summary.losses += 1
                    self._summary.episodes += 1
                    self._summary.episode_lengths.append(episode_steps)
                    self._summary.raw_episode_rewards.append(raw_episode_reward)
                    self._summary.training_episode_rewards.append(train_episode_reward)
                    record = {
                        **episode_metadata,
                        "env_step": env_step,
                        "episode": self._summary.episodes,
                        "result": last_result,
                        "episode_steps": episode_steps,
                        "episode_ticks": int(next_info.get("episode_ticks", 0)),
                        "window_start_game_clock": next_info.get("window_start_game_clock"),
                        "game_clock": next_info.get("game_clock"),
                        "tick_clock_rebased": bool(next_info.get("tick_clock_rebased", False)),
                        "max_episode_ticks": self.config.env.max_episode_ticks,
                        "truncated": bool(truncated),
                        "truncation_reason": next_info.get("truncation_reason"),
                        "raw_episode_reward": raw_episode_reward,
                        "training_episode_reward": train_episode_reward,
                        "reward_breakdown": next_info.get("reward_breakdown"),
                        "epsilon": self._summary.epsilon,
                        "replay_size": len(replay),
                        "loss": self._summary.last_loss,
                        "invalid_actions": episode_invalid,
                        "mowers_used": episode_mowers,
                        "first_mower_step": credit.first_mower_step,
                        "first_mower_tick": credit.first_mower_tick,
                        "credit_assignment": self.config.credit_assignment,
                        "win_rate": self._summary.win_rate,
                        "mean_ticks_advanced": self._summary.mean_ticks_advanced,
                        "std_ticks_advanced": self._summary.std_ticks_advanced,
                        "tick_samples": self._summary.tick_samples,
                        "long_tick_jumps": self._summary.long_tick_jumps,
                        "long_tick_jump_rate": self._summary.long_tick_jump_rate,
                    }
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                    self._emit("episode_end", **record)
                    self._record_curriculum_episode(record)

                    evaluation_due = (
                        self._is_evaluation_due()
                        and not self.stop_event.is_set()
                    )
                    if evaluation_due:
                        try:
                            self._evaluation_active = True
                            eval_record = timing.measure("evaluation", self._evaluate_current_agent,
                                env,
                                action_dim,
                                env_step=env_step,
                            )
                            self._record_evaluation(eval_record, evaluations_path)
                        except Exception as exc:
                            # Evaluation is an auxiliary measurement path. A
                            # serialization/metric failure must not discard a
                            # healthy training run; the following training reset
                            # gets a chance to recover the environment.
                            self._emit(
                                "evaluation_error",
                                env_step=env_step,
                                episode=self._summary.episodes,
                                error=f"{type(exc).__name__}: {exc}",
                            )
                        finally:
                            self._evaluation_active = False

                    if env_step < self.config.total_steps and not self.stop_event.is_set():
                        if getattr(self.config.env, "level_pool", None):
                            cur_lvl, cur_p = self._curriculum.sample(self._episode_rng)
                            env.level_id = cur_lvl
                            env.playthrough = cur_p
                        state, info = timing.measure("reset", env.reset, seed=_episode_seed(self._episode_rng))
                        episode_metadata = episode_level_metadata(info, self.config.env, env)
                        self._emit("seed_selection", **info.get("seed_selection", {}))
                        state = normalize_observation(state)
                        mask = _require_mask(info, action_dim)
                        credit = EpisodeEventCredit(replay, self.config, info)
                        state = credit.observe(state)
                        episode_steps = 0
                        raw_episode_reward = 0.0
                        train_episode_reward = 0.0
                        episode_mowers = 0
                        episode_invalid = 0
                        last_result = "UNKNOWN"
                        previous_replay_eligible = bool(info.get("replay_eligible", True))
                        visual_episode += 1
                        bind_visualization(info)

            if self._summary.stopped or self.stop_event.is_set():
                self._summary.stopped = True
                self._state = "STOPPED"
                if self._agent is not None:
                    stopped = timing.measure("save", self.save_checkpoint, self.run_dir / "stopped.pt")
                    timing.measure("save_sync", self._sync_latest_checkpoint, stopped)
            else:
                self._state = "FINISHED"
                if self._agent is not None:
                    final = timing.measure("save", self.save_checkpoint, self.run_dir / "final.pt")
                    timing.measure("save_sync", self._sync_latest_checkpoint, final)
            self._emit("finished", state=self._state)
            return self._summary
        except Exception as exc:
            self._state = "FAILED"
            self._emit("error", error=str(exc), state=self._state)
            raise
        finally:
            try:
                self._flush_checkpoint_writer(close=True)
                timing.save(self.run_dir / "timing.json")
                bridge_timing.save(self.run_dir / "bridge_timing.json")
            except OSError:
                logger.warning("Could not save timing report", exc_info=True)
            self._visualization.clear()
            try:
                env.close()
            except Exception:
                pass
            self._env = None
            self._runtime_reset_hook = None


__all__ = ["DoubleDQNSummary", "DoubleDQNTrainerController", "EventCallback"]
