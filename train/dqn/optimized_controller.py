"""Performance-oriented Double DQN controller policy.

The battle/training algorithm remains in ``base_controller.py``. This layer
changes runtime infrastructure policy only:

* frequent ``latest.pt`` autosaves are model-only;
* replay snapshots are written only by full/manual/final checkpoints;
* default evaluation runs four games per training level/playthrough every 400 episodes;
* checkpoints can resume a run or initialize a fresh run with weights and optional paired Replay.

Explicit legacy evaluation arguments remain authoritative for tests/tools that
need one exact protocol.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .agent import DoubleDQNAgent
from .base_controller import DoubleDQNTrainerController as _BaseController
from .replay import ReplayBuffer
from .reward import effective_reward_settings


from .checkpoint_contract import (
    CHECKPOINT_LOAD_RESUME, CHECKPOINT_LOAD_STRICT, CHECKPOINT_LOAD_MODEL,
    CHECKPOINT_LOAD_WEIGHTS_ONLY, CHECKPOINT_LOAD_WEIGHTS_REPLAY, CHECKPOINT_LOAD_MODES,
)


class DoubleDQNTrainerController(_BaseController):
    """Drop-in controller with lower runtime overhead and explicit checkpoint load modes."""

    def __init__(
        self,
        *args,
        checkpoint_load_mode: str = CHECKPOINT_LOAD_RESUME,
        fast_eval_every_episodes: int = 0,
        fast_eval_seed_count: int = 5,
        full_eval_every_episodes: int = 400,
        full_eval_seed_count: int = 20,
        **kwargs,
    ) -> None:
        if fast_eval_every_episodes < 0 or full_eval_every_episodes < 0:
            raise ValueError("evaluation cadence must be non-negative")
        if fast_eval_seed_count <= 0 or full_eval_seed_count <= 0:
            raise ValueError("evaluation seed counts must be positive")
        if checkpoint_load_mode not in CHECKPOINT_LOAD_MODES:
            raise ValueError(
                "checkpoint_load_mode must be one of " + ", ".join(CHECKPOINT_LOAD_MODES)
            )

        self.checkpoint_load_mode = checkpoint_load_mode

        # If a caller explicitly supplied the base evaluation knobs, preserve
        # that exact protocol instead of silently converting it into tiered eval.
        self._tiered_eval = not any(
            name in kwargs
            for name in (
                "eval_every_episodes",
                "eval_seed_start",
                "eval_seed_count",
                "eval_max_episode_steps",
            )
        )
        self.fast_eval_every_episodes = int(fast_eval_every_episodes)
        self.fast_eval_seed_count = int(fast_eval_seed_count)
        self.full_eval_every_episodes = int(full_eval_every_episodes)
        self.full_eval_seed_count = int(full_eval_seed_count)
        self._legacy_evaluation_protocol = (not self._tiered_eval or
            fast_eval_seed_count != 5 or full_eval_seed_count != 20)

        if self._tiered_eval:
            kwargs["eval_every_episodes"] = (
                self.fast_eval_every_episodes or self.full_eval_every_episodes
            )
            kwargs["eval_seed_count"] = (
                self.fast_eval_seed_count if self.fast_eval_every_episodes
                else self.full_eval_seed_count
            )
        super().__init__(*args, **kwargs)
        if checkpoint_load_mode == CHECKPOINT_LOAD_WEIGHTS_REPLAY:
            if self.checkpoint_path is None:
                raise ValueError("weights_replay requires a checkpoint with a paired Replay")
            if self.run_dir.resolve() == self.checkpoint_path.parent.resolve():
                raise ValueError("weights_replay requires an independent new run directory")
            if any((self.run_dir / name).exists() for name in
                   ("config.json", "episodes.jsonl", "evaluations.jsonl")):
                raise ValueError("weights_replay output already contains training history; choose a new run directory")
        self._model_only_autosave = True

    def _build_agent(
        self,
        action_dim: int,
    ) -> tuple[DoubleDQNAgent, int, int, dict[str, Any]]:
        if self.checkpoint_path is None or self.checkpoint_load_mode not in (
                CHECKPOINT_LOAD_WEIGHTS_ONLY, CHECKPOINT_LOAD_WEIGHTS_REPLAY):
            return super()._build_agent(action_dim)

        # Both transfer modes retain online weights and reset training state at step 0.
        # weights_replay separately verifies and restores the source experience.
        agent, checkpoint_config, payload = DoubleDQNAgent.from_checkpoint(
            self.checkpoint_path,
            device=self.config.device,
            weights_only=True,
            boss_projectiles=self.config.boss_projectiles,
        )
        if agent.action_dim != action_dim:
            raise RuntimeError(
                f"checkpoint action dimension {agent.action_dim} does not match "
                f"environment {action_dim}"
            )
        if self.checkpoint_load_mode == CHECKPOINT_LOAD_WEIGHTS_REPLAY:
            from .checkpoint_contract import inspect_checkpoint
            self._checkpoint_kind = inspect_checkpoint(payload)
            if self._checkpoint_kind == "model_only":
                raise ValueError("weights_replay requires a paired Replay; checkpoint is model-only")
            replay_file = payload.get("replay_file", "replay.npz")
            if not isinstance(replay_file, str) or Path(replay_file).name != replay_file:
                raise ValueError("invalid checkpoint replay_file")
            self._checkpoint_replay_file = self.checkpoint_path.parent / replay_file
            self._checkpoint_reward_settings = effective_reward_settings(checkpoint_config)
            self._checkpoint_observation_migration = payload.get("observation_migration")
            self._source_replay_progress = (int(payload.get("env_step", 0)),
                                           int(payload.get("episode", 0)))
            self._emit("checkpoint_restore_contract", path=str(self.checkpoint_path),
                       checkpoint_kind=self._checkpoint_kind, load_mode=self.checkpoint_load_mode,
                       exact_game_resume=False)
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - from_checkpoint already guards this.
            raise RuntimeError("Double DQN requires PyTorch") from exc

        self.config = replace(self.config, hidden_sizes=agent.hidden_sizes,
                              credit_assignment=agent.credit_assignment)
        agent.set_reward_config(self.config)
        agent._rng = np.random.default_rng(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        # The online network is the knowledge being transferred. A fresh target
        # network starts synchronized to it, while optimizer momentum/update
        # counters are deliberately discarded.
        agent.target.load_state_dict(agent.online.state_dict())
        agent.target.eval()
        agent.optimizer = torch.optim.Adam(
            agent.online.parameters(),
            lr=self.config.learning_rate,
        )
        agent.update_count = 0
        self._episode_rng = np.random.default_rng(self.config.seed)
        agent.online.train()

        self._latest_checkpoint = self.checkpoint_path
        self._summary.checkpoint = str(self.checkpoint_path)
        # Worker progress belongs to exact resume, not transferred weights.
        return agent, 0, 0, {key: value for key, value in payload.items() if key != "sampling_state"}

    def _restore_replay(
        self,
        *,
        action_dim: int,
        start_step: int,
        start_episode: int,
    ) -> ReplayBuffer:
        if self.checkpoint_load_mode == CHECKPOINT_LOAD_WEIGHTS_REPLAY:
            replay = super()._restore_replay(
                action_dim=action_dim, start_step=self._source_replay_progress[0],
                start_episode=self._source_replay_progress[1])
            # The historical games are interrupted. New workers must never
            # link their first transition to an unfinished historical n-step chain.
            for stream_id in tuple(replay._stream_tails):
                replay.end_episode(stream_id)
            replay._rng = np.random.default_rng(self.config.seed + 1)
            return replay
        if self.checkpoint_load_mode != CHECKPOINT_LOAD_WEIGHTS_ONLY:
            return super()._restore_replay(
                action_dim=action_dim,
                start_step=start_step,
                start_episode=start_episode,
            )

        self._summary.replay_restored = False
        return self._new_replay(action_dim)

    def save_checkpoint(
        self,
        path: str | Path | None = None,
        *,
        _internal: bool = False,
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
        if callable(getattr(agent, 'checkpoint_payload', None)):
            saved = super().save_checkpoint(destination, _internal=_internal)
            self._emit('checkpoint', path=str(saved),
                kind='model_only' if destination.name in ('latest.pt', 'latest-model.pt') else 'full')
            return saved

        model_started = time.perf_counter()
        with self._agent_lock:
            saved = agent.save(
                destination,
                env_step=env_step,
                episode=episode,
                training_summary=self._summary.snapshot(),
                episode_rng_state=self._episode_rng.bit_generator.state,
                seed_types=self.seed_types,
            )
        model_save_ms = (time.perf_counter() - model_started) * 1000.0

        # latest.pt is the frequent autosave. Replay compression is reserved for
        # milestone/manual/final checkpoints.
        save_replay = destination.name != "latest.pt"
        replay_save_ms = 0.0
        replay = self._replay
        if save_replay and replay is not None:
            replay_started = time.perf_counter()
            with self._replay_lock:
                replay_path = replay.save(
                    self._replay_path(),
                    env_step=env_step,
                    episode=episode,
                    gamma=self.config.gamma,
                    gamma_reference_ticks=self.config.gamma_reference_ticks,
                    reward_settings=effective_reward_settings(self.config),
                )
            replay_save_ms = (time.perf_counter() - replay_started) * 1000.0
            self._emit(
                "replay_checkpoint",
                path=str(replay_path),
                env_step=env_step,
                replay_size=len(replay),
                replay_save_ms=replay_save_ms,
            )

        self._latest_checkpoint = saved
        self._summary.checkpoint = str(saved)
        self._emit(
            "checkpoint",
            path=str(saved),
            kind="model_only" if not save_replay else "full",
            model_save_ms=model_save_ms,
            replay_save_ms=replay_save_ms,
        )
        return saved

    def _is_evaluation_due(self) -> bool:
        if getattr(self.config, "evaluation_mode", None) in ("off", "none"):
            return False
        if not self._tiered_eval:
            return super()._is_evaluation_due()
        return self._is_full_eval_episode() or (
            self.fast_eval_every_episodes > 0
            and self._summary.episodes > 0
            and self._summary.episodes % self.fast_eval_every_episodes == 0
        )

    def _is_full_eval_episode(self) -> bool:
        return (
            self.full_eval_every_episodes > 0
            and self._summary.episodes > 0
            and self._summary.episodes % self.full_eval_every_episodes == 0
        )

    def _evaluate_current_agent(self, env, action_dim: int, *, env_step: int) -> dict[str, Any]:
        if not self._tiered_eval:
            started = time.perf_counter()
            record = super()._evaluate_current_agent(env, action_dim, env_step=env_step)
            record["kind"] = "full"
            record["evaluation_wall_time_ms"] = (time.perf_counter() - started) * 1000.0
            return record

        full = self._is_full_eval_episode()
        previous_count = self.eval_seed_count
        self.eval_seed_count = self.full_eval_seed_count if full else self.fast_eval_seed_count
        started = time.perf_counter()
        try:
            record = super()._evaluate_current_agent(env, action_dim, env_step=env_step)
        finally:
            self.eval_seed_count = previous_count
        record["kind"] = "full" if full else "fast"
        record["evaluation_wall_time_ms"] = (time.perf_counter() - started) * 1000.0
        self._emit(
            "evaluation_profile",
            kind=record["kind"],
            env_step=env_step,
            episode=self._summary.episodes,
            evaluation_wall_time_ms=record["evaluation_wall_time_ms"],
        )
        return record

    def _evaluation_protocol_matches(self, record: dict[str, Any]) -> bool:
        from .evaluation_plan import protocol_matches
        count = self.full_eval_seed_count if self._tiered_eval else self.eval_seed_count
        return str(record.get("kind", "full")) == "full" and protocol_matches(self, record, count)

    def _record_evaluation(self, record: dict[str, Any], evaluations_path: Path) -> None:
        if not self._tiered_eval or str(record.get("kind", "fast")) == "full":
            super()._record_evaluation(record, evaluations_path)
            return

        # Fast evaluations are trend diagnostics only and never replace best.pt.
        with evaluations_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


__all__ = [
    "CHECKPOINT_LOAD_MODES",
    "CHECKPOINT_LOAD_RESUME",
    "CHECKPOINT_LOAD_WEIGHTS_ONLY",
    "DoubleDQNTrainerController",
]
