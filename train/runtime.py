"""Shared runtime configuration and sampling/inference/learning contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from pvz_gym.runtime import RuntimeCapabilityError


@dataclass(frozen=True)
class TrainingRuntimeConfig:
    num_envs: int = 1
    asynchronous: bool = False
    queue_capacity: int = 1024
    inference_batch_size: int = 1
    inference_max_wait_ms: float = 2.0
    process_ids: tuple[int, ...] = ()
    game_executable: str | None = None
    instance_root: str | None = None
    save_template: str | None = None
    save_directory: str | None = None
    profile_id: int | None = None
    visualize_q: bool = False
    game_topmost: bool = False
    worker_timeout_s: float = 60.0
    restart_limit: int = 2
    max_policy_lag: int = 64
    ui_update_interval_ms: int = 200
    asynchronous_checkpoints: bool = True
    max_episodes_per_worker: int = 0

    def __post_init__(self) -> None:
        if type(self.visualize_q) is not bool or type(self.game_topmost) is not bool:
            raise ValueError("visualize_q and game_topmost must be booleans")
        if self.visualize_q and self.num_envs != 1:
            raise ValueError("Q overlay requires one selected training environment")
        for name in ("instance_root", "save_template", "save_directory"):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a path string or None")
        if self.profile_id is not None and (type(self.profile_id) is not int or self.profile_id <= 0):
            raise ValueError("profile_id must be a positive integer or None")
        object.__setattr__(self, "process_ids", tuple(self.process_ids))
        for name in ("num_envs", "queue_capacity", "inference_batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.asynchronous) is not bool:
            raise ValueError("asynchronous must be a boolean")
        if type(self.asynchronous_checkpoints) is not bool:
            raise ValueError("asynchronous_checkpoints must be a boolean")
        if (type(self.inference_max_wait_ms) not in (int, float)
                or not math.isfinite(self.inference_max_wait_ms)
                or self.inference_max_wait_ms < 0):
            raise ValueError("inference_max_wait_ms must be finite and non-negative")
        if self.process_ids and (len(self.process_ids) != self.num_envs or len(set(self.process_ids)) != self.num_envs
                or any(type(pid) is not int or not 0 < pid < 2**32 for pid in self.process_ids)):
            raise ValueError("process_ids must contain one distinct positive PID per environment")
        for name in ("restart_limit", "max_policy_lag", "ui_update_interval_ms", "max_episodes_per_worker"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not math.isfinite(self.worker_timeout_s) or self.worker_timeout_s <= 0:
            raise ValueError("worker_timeout_s must be finite and positive")
        if self.game_executable is not None and not isinstance(self.game_executable, str):
            raise ValueError("game_executable must be a path string")


def require_serial_training(config: TrainingRuntimeConfig) -> None:
    """Guard trainers that do not use the DQN process-sampler controller."""
    missing = []
    if config.num_envs != 1:
        missing.append("multi-environment sampling")
    if config.asynchronous:
        missing.append("asynchronous sampling/learning")
    if config.inference_batch_size != 1:
        missing.append("batched inference")
    if missing:
        raise RuntimeCapabilityError("this trainer supports serial sampling only: " + ", ".join(missing))


def uses_parallel_runtime(config: TrainingRuntimeConfig) -> bool:
    return config.num_envs > 1 or config.asynchronous or config.inference_batch_size > 1 or bool(config.game_executable or config.process_ids)


@dataclass(frozen=True)
class EnvironmentBinding:
    environment_id: int
    process_id: int
    seed: int


@dataclass(frozen=True)
class SampleIdentity:
    """A reply is valid only for the matching env, episode and decision."""

    environment_id: int
    episode_id: int
    sequence_id: int


@dataclass(frozen=True)
class PolicyRequest:
    identity: SampleIdentity
    observation: Mapping[str, object]
    action_mask: tuple[bool, ...]
    minimum_policy_version: int = 0


@dataclass(frozen=True)
class PolicyReply:
    identity: SampleIdentity
    action: int
    policy_version: int


@dataclass(frozen=True)
class TransitionEnvelope:
    """Queue messages own their payloads; do not reuse mutable snapshot buffers.

    Use actual battle ticks for discounting. N-step/reward histories are keyed by
    (environment_id, episode_id), and both terminal and truncated flush them.
    """

    identity: SampleIdentity
    policy_version: int
    observation: Mapping[str, object]
    next_observation: Mapping[str, object]
    action: int
    reward: float
    action_mask: tuple[bool, ...]
    next_action_mask: tuple[bool, ...]
    ticks_advanced: int
    terminated: bool
    truncated: bool
    info: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicySnapshot:
    version: int
    state: object


class Sampler(Protocol):
    """Bounded collection with explicit per-environment episode ownership."""

    def start(self, restored: Mapping[str, object] | None = None) -> None: ...
    def receive(self, timeout: float = 0.05) -> tuple[str, int, object] | None: ...
    def dispatch(self, reply: PolicyReply) -> None: ...
    def reset(self, environment_id: int, seed: int | None = None) -> None: ...
    def restart(self, environment_id: int) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def close(self) -> None: ...


class InferenceService(Protocol):
    def publish(self, snapshot: PolicySnapshot) -> None: ...
    def infer(self, requests: Sequence[PolicyRequest]) -> tuple[PolicyReply, ...]: ...
    def close(self) -> None: ...


class Learner(Protocol):
    """Update scheduling counts aggregate valid transitions, not worker loops."""

    def ingest(self, transitions: Sequence[TransitionEnvelope]) -> None: ...
    def update(self, *, gradient_steps: int) -> Mapping[str, float]: ...
    def policy_snapshot(self) -> PolicySnapshot: ...
    def checkpoint_state(self) -> Mapping[str, object]: ...
