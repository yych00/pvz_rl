"""Double DQN 使用的紧凑结构化回放缓存。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pvz_gym.observation import OBSERVATION_SCHEMA, LEGACY_OBSERVATION_SCHEMA
from .config import CREDIT_ASSIGNMENT_MODES, DISCOUNT_SCHEMA_VERSION, convert_gamma
from .reward_observation import (DENSE_OBSERVATION_SCHEMA, LEGACY_DENSE_OBSERVATION_SCHEMA,
                                 observation_schema, observation_spec)


ObservationBatch = dict[str, np.ndarray]


@dataclass(frozen=True)
class ReplayBatch:
    observations: ObservationBatch
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: ObservationBatch
    dones: np.ndarray
    next_action_masks: np.ndarray
    ticks_advanced: np.ndarray
    indices: np.ndarray | None = None
    weights: np.ndarray | None = None


def _storage_dtype(dtype: np.dtype) -> np.dtype:
    if np.issubdtype(dtype, np.floating):
        return np.dtype(np.float16)
    if np.issubdtype(dtype, np.signedinteger):
        return np.dtype(np.uint16) if dtype.itemsize > 1 else dtype
    return dtype


def _sample_dtype(dtype: np.dtype) -> np.dtype:
    if np.issubdtype(dtype, np.floating):
        return np.dtype(np.float32)
    if dtype.itemsize > 1:
        return np.dtype(np.int64)
    return dtype


class ReplayBuffer:
    """固定容量的循环缓存，完整保留结构化观测中的每个字段。"""

    def __init__(self, capacity: int, action_dim: int, *, seed: int = 0,
                 credit_assignment: str = "one_step",
                 scheme: str = "uniform",
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 eps: float = 1e-5, reward_context: bool = False, boss_projectiles: bool = True):
        if capacity <= 0 or action_dim <= 0:
            raise ValueError("capacity and action_dim must be positive")
        if credit_assignment not in CREDIT_ASSIGNMENT_MODES:
            raise ValueError("unknown replay credit_assignment")
        from .sum_tree import SumTree
        self.scheme = str(scheme)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)
        self.max_priority = 1.0
        # Keep unexponentiated priorities so alpha=0 does not erase TD history.
        self._priorities = np.ones(capacity, dtype=np.float64)
        self._tree: SumTree | None = SumTree(capacity) if self.scheme == "per" else None
        self.credit_assignment = credit_assignment
        self.next_transition_id = 0
        self.capacity = int(capacity)
        self.action_dim = int(action_dim)
        self.mask_bytes = (self.action_dim + 7) // 8
        self.boss_projectiles = bool(boss_projectiles)
        self.observation_schema = observation_schema(reward_context, self.boss_projectiles)
        self.observation_spec = observation_spec(reward_context, self.boss_projectiles)
        self._rng = np.random.default_rng(seed)
        self._position = 0
        self._size = 0
        self._observations = {
            name: np.empty((capacity, *shape), dtype=_storage_dtype(dtype))
            for name, (shape, dtype) in self.observation_spec.items()
        }
        self._next_observations = {
            name: np.empty((capacity, *shape), dtype=_storage_dtype(dtype))
            for name, (shape, dtype) in self.observation_spec.items()
        }
        self._actions = np.empty(capacity, dtype=np.int32)
        self._rewards = np.empty(capacity, dtype=np.float32)
        self._dones = np.empty(capacity, dtype=np.bool_)
        self._episode_ends = np.empty(capacity, dtype=np.bool_)
        self.supports_n_step = True
        self.reward_settings: dict[str, Any] | None = None
        self._ticks_advanced = np.empty(capacity, dtype=np.int64)
        self._next_action_masks = np.empty((capacity, self.mask_bytes), dtype=np.uint8)
        self._stream_ids = np.full(capacity, -1, dtype=np.int32)
        self._transition_ids = np.full(capacity, -1, dtype=np.int64)
        self._next_ids = np.full(capacity, -1, dtype=np.int64)
        self._previous_ids = np.full(capacity, -1, dtype=np.int64)
        self._stream_tails: dict[int, int] = {}
        self._eligible_cache: dict[int, np.ndarray] = {}
        self._eligible_counts: dict[int, int] = {}
        self.set_scheme(self.scheme)

    def set_reward_context(self, enabled):
        schema = observation_schema(enabled, self.boss_projectiles)
        if schema == self.observation_schema:
            return
        if self._size:
            raise ValueError("reward observation context changed; rebuild replay")
        self.observation_schema = schema
        self.observation_spec = observation_spec(enabled, self.boss_projectiles)
        for arrays in (self._observations, self._next_observations):
            if enabled:
                arrays["reward_context"] = np.empty((self.capacity, 3), dtype=np.float16)
            else:
                arrays.pop("reward_context", None)

    def __len__(self) -> int:
        return self._size

    def _validate(self, observation: object) -> None:
        if not isinstance(observation, dict) or set(observation) != set(self.observation_spec):
            raise ValueError("observation does not match replay schema")
        for name, (shape, _dtype) in self.observation_spec.items():
            if np.asarray(observation[name]).shape != shape:
                raise ValueError(f"observation field {name} has the wrong shape")

    def add(
        self,
        observation: ObservationBatch,
        action: int,
        reward: float,
        next_observation: ObservationBatch,
        done: bool,
        next_action_mask,
        *,
        ticks_advanced: int,
        episode_end: bool | None = None,
        stream_id: int = 0,
    ) -> int:
        self._validate(observation)
        self._validate(next_observation)
        if type(ticks_advanced) is not int or not 0 <= ticks_advanced < 2**63:
            raise ValueError("ticks_advanced must be a non-negative int64 integer")
        mask = np.asarray(next_action_mask, dtype=np.bool_)
        if mask.shape != (self.action_dim,):
            raise ValueError("next action mask shape does not match replay buffer")
        if not 0 <= int(action) < self.action_dim:
            raise ValueError("action outside replay action domain")

        index = self._position
        if type(stream_id) is not int or not 0 <= stream_id < 2**31:
            raise ValueError("stream_id must be a non-negative int32")
        for name in self.observation_spec:
            self._observations[name][index] = observation[name]
            self._next_observations[name][index] = next_observation[name]
        self._actions[index] = int(action)
        self._rewards[index] = float(reward)
        self._dones[index] = bool(done)
        self._episode_ends[index] = bool(done or episode_end)
        self._ticks_advanced[index] = ticks_advanced
        self._next_action_masks[index] = np.packbits(mask, bitorder="little")
        transition_id = self.next_transition_id
        previous_id = self._stream_tails.get(stream_id, -1)
        previous = self._slot_for_id(previous_id)
        if previous is not None and not self._episode_ends[previous]:
            self._next_ids[previous] = transition_id
        else:
            previous_id = -1
        self._stream_ids[index] = stream_id
        self._transition_ids[index] = transition_id
        self._next_ids[index] = -1
        self._previous_ids[index] = previous_id
        self._stream_tails[stream_id] = transition_id
        self.next_transition_id += 1
        if self._tree is not None:
            self._priorities[index] = self.max_priority
            self._tree.update(index, self.max_priority ** self.alpha)
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._refresh_eligibility(transition_id)
        return transition_id

    def _slot_for_id(self, transition_id: int) -> int | None:
        if transition_id < 0: return None
        # Loading may compact a ring, so base the slot on its current tail.
        offset = transition_id - (self.next_transition_id - self._size)
        if not 0 <= offset < self._size: return None
        slot = (self._position - self._size + offset) % self.capacity
        return slot if self._transition_ids[slot] == transition_id else None

    def _complete_return(self, slot: int, n_step: int) -> bool:
        for step in range(n_step):
            if self._episode_ends[slot] or step == n_step-1: return True
            slot = self._slot_for_id(int(self._next_ids[slot]))
            if slot is None: return False
        return True

    def _refresh_eligibility(self, transition_id: int) -> None:
        for n_step, cache in self._eligible_cache.items():
            slot = self._slot_for_id(transition_id)
            for _ in range(n_step):
                if slot is None: break
                complete = self._complete_return(slot, n_step)
                self._eligible_counts[n_step] += int(complete) - int(cache[slot])
                cache[slot] = complete
                slot = self._slot_for_id(int(self._previous_ids[slot]))

    def _ensure_eligibility(self, n_step: int) -> np.ndarray:
        if type(n_step) is not int or not 1 <= n_step <= self.capacity:
            raise ValueError("n_step must be an integer in [1, replay capacity]")
        if n_step > 1 and not self.supports_n_step:
            raise ValueError("legacy replay lacks episode boundaries for n-step returns")
        if n_step not in self._eligible_cache:
            cache = np.zeros(self.capacity, dtype=np.bool_)
            for slot in range(self._size): cache[slot] = self._complete_return(slot, n_step)
            self._eligible_cache[n_step] = cache
            self._eligible_counts[n_step] = int(np.count_nonzero(cache))
        return self._eligible_cache[n_step]

    def _eligible_indices(self, n_step: int) -> np.ndarray:
        cache = self._ensure_eligibility(n_step)
        order = (self._position - self._size + np.arange(self._size)) % self.capacity
        return order[cache[order]]

    @property
    def is_per(self) -> bool:
        return self.scheme == "per" and self._tree is not None

    def set_scheme(
        self,
        scheme: str,
        *,
        alpha: float | None = None,
        beta: float | None = None,
        eps: float | None = None,
    ) -> None:
        if scheme not in ("uniform", "per"):
            raise ValueError(f"unknown replay scheme: {scheme}")
        new_alpha = self.alpha if alpha is None else float(alpha)
        new_beta = self.beta if beta is None else float(beta)
        new_eps = self.eps if eps is None else float(eps)
        if not (math.isfinite(new_alpha) and 0 <= new_alpha <= 1
                and math.isfinite(new_beta) and 0 <= new_beta <= 1
                and math.isfinite(new_eps) and 0 < new_eps <= 1e5):
            raise ValueError("invalid PER parameters")
        rebuild = new_alpha != self.alpha or new_eps != self.eps
        if new_eps != self.eps:
            self._priorities[:self._size] = np.clip(
                self._priorities[:self._size] + new_eps - self.eps, new_eps, 1e5)
            self.max_priority = float(np.clip(self.max_priority + new_eps - self.eps, new_eps, 1e5))
        if alpha is not None:
            self.alpha = float(alpha)
        if beta is not None:
            self.beta = float(beta)
        if eps is not None:
            self.eps = float(eps)

        if scheme == "per":
            if self._tree is None:
                from .sum_tree import SumTree
                self._tree = SumTree(self.capacity)
                self._priorities[:self._size] = self.max_priority
                rebuild = True
        if self._tree is not None and (rebuild or self._tree.total <= 0):
            self._tree.set_all(np.power(self._priorities[:self._size], self.alpha), self._size)
        self.scheme = scheme

    def update_priorities(self, indices: Sequence[int] | np.ndarray, priorities: Sequence[float] | np.ndarray) -> None:
        if self._tree is None:
            return
        idx_arr = np.asarray(indices, dtype=np.int64)
        prios = np.asarray(priorities, dtype=np.float64)
        if idx_arr.shape != prios.shape:
            raise ValueError("indices and priorities must have matching shapes")
        if not idx_arr.size:
            return
        if (idx_arr < 0).any() or (idx_arr >= self._size).any():
            raise IndexError("priority index outside active replay")
        if not np.isfinite(prios).all():
            raise ValueError("priorities must be finite")
        prios = np.abs(prios) + self.eps
        prios = np.clip(prios, self.eps, 1e5)
        current_max = float(np.max(prios))
        if current_max > self.max_priority:
            self.max_priority = current_max
        p_alpha = np.power(prios, self.alpha)
        self._priorities[idx_arr] = prios
        self._tree.update_batch(idx_arr, p_alpha)

    def end_episode(self, stream_id: int = 0) -> None:
        """Seal a reset/interruption boundary without inventing a terminal reward."""
        slot = self._slot_for_id(self._stream_tails.get(stream_id, -1))
        if slot is not None:
            self._episode_ends[slot] = True
            self._refresh_eligibility(int(self._transition_ids[slot]))

    def _eligible_count(self, n_step: int) -> int:
        self._ensure_eligibility(n_step)
        return self._eligible_counts[n_step]

    def can_sample(self, n_step: int = 1) -> bool:
        return self._eligible_count(n_step) > 0

    def sample(self, batch_size: int, *, n_step: int = 1, gamma: float = 0.999,
               gamma_reference_ticks: int = 100, beta: float | None = None) -> ReplayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._size < batch_size:
            raise ValueError(f"not enough replay entries: {self._size} < {batch_size}")
        eligible = self._eligible_count(n_step)
        if not eligible:
            raise ValueError("no complete n-step returns in replay yet")
        convert_gamma(gamma, gamma_reference_ticks, gamma_reference_ticks)
        if self.scheme == "per" and self._tree is not None and self._tree.total > 0:
            current_beta = self.beta if beta is None else float(beta)
            # Active ring slots occupy [0, size) until full. Only incomplete
            # stream tails need excluding; avoid full-size int64 index arrays.
            excluded = np.flatnonzero(~self._eligible_cache[n_step][:self._size])
            if excluded.size:
                excluded_priorities = self._tree.get_priorities(excluded)
                self._tree.update_batch(excluded, np.zeros(excluded.size))
            else:
                excluded_priorities = None
            try:
                total = self._tree.total
                segment = total / batch_size
                targets = self._rng.uniform(np.arange(batch_size) * segment,
                                            (np.arange(batch_size) + 1) * segment)
                sampled_indices, stored_p_alpha = self._tree.sample(targets, active_size=self._size)
            finally:
                if excluded.size and excluded_priorities is not None:
                    self._tree.update_batch(excluded, excluded_priorities)
            indices = sampled_indices
            probs = np.maximum(stored_p_alpha / total, 1e-12)
            n_samples = float(eligible)
            is_w = np.power(n_samples * probs, -current_beta)
            max_w = float(np.max(is_w))
            is_weights = (is_w / max_w).astype(np.float32) if max_w > 0 else np.ones(batch_size, dtype=np.float32)
        else:
            eligible_indices = self._eligible_indices(n_step)
            indices = eligible_indices[self._rng.integers(0, eligible, size=batch_size)]
            is_weights = np.ones(batch_size, dtype=np.float32)

        endpoints = indices.copy()
        returns = np.zeros(batch_size, dtype=np.float64)
        elapsed = np.zeros(batch_size, dtype=np.int64)
        active = np.ones(batch_size, dtype=np.bool_)
        for step in range(n_step):
            rows = np.flatnonzero(active)
            if not rows.size:
                break
            positions = endpoints[rows]
            weights = np.power(gamma, elapsed[rows].astype(np.float64) / gamma_reference_ticks)
            returns[rows] += weights * self._rewards[positions]
            durations = self._ticks_advanced[positions]
            if (elapsed[rows] > np.iinfo(np.int64).max - durations).any():
                raise ValueError("n-step tick duration overflow")
            elapsed[rows] += durations
            endpoints[rows] = positions
            active[rows] = ~self._episode_ends[positions]
            if step + 1 < n_step:
                for b in rows[active[rows]]:
                    successor = self._slot_for_id(int(self._next_ids[endpoints[b]]))
                    if successor is None: raise RuntimeError("incomplete replay stream return")
                    endpoints[b] = successor
        observations = {
            name: values[indices].astype(_sample_dtype(self.observation_spec[name][1]))
            for name, values in self._observations.items()
        }
        next_observations = {
            name: values[endpoints].astype(_sample_dtype(self.observation_spec[name][1]))
            for name, values in self._next_observations.items()
        }
        return ReplayBatch(
            observations=observations,
            actions=self._actions[indices].copy(),
            rewards=returns.astype(np.float32),
            next_observations=next_observations,
            dones=self._dones[endpoints].copy(),
            ticks_advanced=elapsed,
            next_action_masks=np.unpackbits(
                self._next_action_masks[endpoints],
                axis=1,
                count=self.action_dim,
                bitorder="little",
            ).astype(np.bool_, copy=False),
            indices=indices.copy(),
            weights=is_weights,
        )

    def checkpoint_payload(self, *, env_step: int = 0, episode: int = 0,
             gamma: float | None = None, gamma_reference_ticks: int | None = None,
             reward_settings: dict[str, Any] | None = None, copy: bool = True) -> dict[str, np.ndarray]:
        if (gamma is None) != (gamma_reference_ticks is None):
            raise ValueError("save both gamma and gamma_reference_ticks together")
        if gamma is not None:
            convert_gamma(gamma, gamma_reference_ticks, gamma_reference_ticks)
        used = self.capacity if self._size == self.capacity else self._size
        payload: dict[str, np.ndarray] = {
            "actions": self._actions[:used],
            "rewards": self._rewards[:used],
            "dones": self._dones[:used],
            "episode_ends": self._episode_ends[:used],
            "raw_return_schema_version": np.asarray(1, dtype=np.int64),
            "supports_n_step": np.asarray(self.supports_n_step, dtype=np.bool_),
            "ticks_advanced": self._ticks_advanced[:used],
            "next_action_masks_packed": self._next_action_masks[:used],
            "capacity": np.asarray(self.capacity, dtype=np.int64),
            "action_dim": np.asarray(self.action_dim, dtype=np.int64),
            "size": np.asarray(self._size, dtype=np.int64),
            "position": np.asarray(self._position, dtype=np.int64),
            "env_step": np.asarray(max(0, int(env_step)), dtype=np.int64),
            "episode": np.asarray(max(0, int(episode)), dtype=np.int64),
            "observation_schema": np.asarray(self.observation_schema),
            "rng_state_json": np.asarray(json.dumps(self._rng.bit_generator.state, separators=(",", ":"))),
            "credit_assignment": np.asarray(self.credit_assignment),
            "next_transition_id": np.asarray(self.next_transition_id, dtype=np.int64),
            "discount_schema_version": np.asarray(DISCOUNT_SCHEMA_VERSION, dtype=np.int64),
            "stream_ids": self._stream_ids[:used],
            "transition_ids": self._transition_ids[:used],
            "next_ids": self._next_ids[:used],
            "stream_schema_version": np.asarray(1, dtype=np.int64),
        }
        if self.scheme == "per" and self._tree is not None:
            payload["replay_scheme"] = np.asarray("per")
            payload["per_alpha"] = np.asarray(self.alpha, dtype=np.float64)
            payload["per_beta"] = np.asarray(self.beta, dtype=np.float64)
            payload["per_eps"] = np.asarray(self.eps, dtype=np.float64)
            payload["per_max_priority"] = np.asarray(self.max_priority, dtype=np.float64)
            payload["per_priorities"] = self._tree.get_priorities(np.arange(used))
            payload["per_raw_priorities"] = self._priorities[:used]
        if gamma is not None and gamma_reference_ticks is not None:
            payload["discount_gamma"] = np.asarray(gamma, dtype=np.float64)
            payload["gamma_reference_ticks"] = np.asarray(gamma_reference_ticks, dtype=np.int64)
        settings = self.reward_settings if reward_settings is None else reward_settings
        if settings is not None:
            payload["reward_settings_json"] = np.asarray(json.dumps(settings, sort_keys=True, allow_nan=False))
        for name in self.observation_spec:
            payload[f"obs__{name}"] = self._observations[name][:used]
            payload[f"next_obs__{name}"] = self._next_observations[name][:used]
        return {name: value.copy() for name, value in payload.items()} if copy else payload

    @staticmethod
    def write_checkpoint_payload(path, payload) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            with temporary.open("wb") as handle: np.savez_compressed(handle, **payload)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def save(self, path: str | Path, **metadata) -> Path:
        return self.write_checkpoint_payload(path, self.checkpoint_payload(copy=False, **metadata))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        capacity: int | None = None,
        action_dim: int | None = None,
        seed: int = 0,
        credit_assignment: str | None = None,
        strict_resume: bool = False,
    ) -> tuple["ReplayBuffer", dict[str, Any]]:
        source = Path(path)
        with np.load(source, allow_pickle=False) as payload:
            if strict_resume:
                required = ("stream_ids", "transition_ids", "next_ids", "stream_schema_version",
                            "raw_return_schema_version", "supports_n_step", "episode_ends",
                            "reward_settings_json", "discount_gamma", "gamma_reference_ticks",
                            "rng_state_json")
                missing = [name for name in required if name not in payload]
                if missing:
                    raise ValueError("strict_resume is missing Replay state: " + ", ".join(missing))
                if str(payload.get("replay_scheme", np.asarray("uniform")).item()) == "per":
                    per_required = ("per_alpha", "per_beta", "per_eps", "per_max_priority",
                                    "per_priorities", "per_raw_priorities")
                    if any(name not in payload for name in per_required):
                        raise ValueError("strict_resume is missing PER priority state")
            version = np.asarray(payload.get("discount_schema_version", DISCOUNT_SCHEMA_VERSION))
            if version.shape != () or version.dtype.kind not in "iu" or int(version) != DISCOUNT_SCHEMA_VERSION:
                raise ValueError("unsupported replay discount_schema_version")
            schema = str(payload["observation_schema"].item())
            if schema not in (OBSERVATION_SCHEMA, DENSE_OBSERVATION_SCHEMA, LEGACY_OBSERVATION_SCHEMA, LEGACY_DENSE_OBSERVATION_SCHEMA):
                raise ValueError(f"replay observation schema {schema!r} is not {OBSERVATION_SCHEMA!r}")
            saved_capacity = int(payload["capacity"])
            saved_action_dim = int(payload["action_dim"])
            saved_size = int(payload["size"])
            saved_position = int(payload["position"])
            saved_credit = str(payload["credit_assignment"].item()) if "credit_assignment" in payload else "one_step"
            if saved_credit not in CREDIT_ASSIGNMENT_MODES:
                raise ValueError("unsupported replay credit_assignment")
            if any(name.startswith("credit__") for name in payload.files):
                raise ValueError("event-credit replay is unsupported; rebuild replay")
            if action_dim is not None and int(action_dim) != saved_action_dim:
                raise ValueError("replay action_dim does not match environment")
            if not 0 <= saved_size <= saved_capacity or not 0 <= saved_position < saved_capacity:
                raise ValueError("invalid replay ring metadata")
            used = saved_capacity if saved_size == saved_capacity else saved_size
            if "ticks_advanced" not in payload:
                raise ValueError("legacy replay lacks actual tick durations; rebuild replay")
            ticks_advanced = np.asarray(payload["ticks_advanced"])
            if (ticks_advanced.shape != (used,) or ticks_advanced.dtype != np.dtype(np.int64)
                    or (ticks_advanced < 0).any()):
                raise ValueError("invalid replay tick durations")
            boss_projectiles = schema in (OBSERVATION_SCHEMA, DENSE_OBSERVATION_SCHEMA)
            dense_context = schema in (DENSE_OBSERVATION_SCHEMA, LEGACY_DENSE_OBSERVATION_SCHEMA)
            spec = observation_spec(dense_context, boss_projectiles)
            actions = np.asarray(payload["actions"], dtype=np.int32)
            rewards = np.asarray(payload["rewards"], dtype=np.float32)
            dones = np.asarray(payload["dones"], dtype=np.bool_)
            raw_keys = ("episode_ends", "raw_return_schema_version", "supports_n_step")
            if any(key in payload for key in raw_keys) and not all(key in payload for key in raw_keys):
                raise ValueError("incomplete raw return replay metadata")
            supports_n_step = False
            episode_ends = dones.copy()
            if all(key in payload for key in raw_keys):
                raw_version = np.asarray(payload["raw_return_schema_version"])
                supported = np.asarray(payload["supports_n_step"])
                episode_ends = np.asarray(payload["episode_ends"])
                if (raw_version.shape != () or raw_version.dtype.kind not in "iu" or int(raw_version) != 1
                        or supported.shape != () or supported.dtype != np.dtype(np.bool_)
                        or episode_ends.shape != (used,) or episode_ends.dtype != np.dtype(np.bool_)
                        or (dones & ~episode_ends).any()):
                    raise ValueError("invalid raw return replay metadata")
                supports_n_step = bool(supported)
            masks = np.asarray(payload["next_action_masks_packed"], dtype=np.uint8)
            if actions.shape != (used,) or rewards.shape != (used,) or dones.shape != (used,):
                raise ValueError("invalid replay transition array shape")
            mask_bytes = (saved_action_dim + 7) // 8
            if masks.shape != (used, mask_bytes):
                raise ValueError("invalid replay action-mask shape")

            target_capacity = int(capacity) if capacity is not None else saved_capacity
            if target_capacity <= 0:
                raise ValueError("replay capacity must be positive")
            if saved_size < saved_capacity:
                order = np.arange(saved_size, dtype=np.int64)
            else:
                order = np.concatenate((
                    np.arange(saved_position, saved_capacity, dtype=np.int64),
                    np.arange(0, saved_position, dtype=np.int64),
                ))
            keep = min(saved_size, target_capacity)
            order = order[-keep:] if keep else order[:0]
            replay = cls(target_capacity, saved_action_dim, seed=seed, credit_assignment=credit_assignment or saved_credit,
                         reward_context=dense_context, boss_projectiles=boss_projectiles)
            replay.supports_n_step = supports_n_step
            # Load one observation field at a time. The chronological retained
            # range crosses the ring boundary at most once, so copy slices
            # directly rather than materializing another reordered field.
            for name, (shape, dtype) in spec.items():
                for prefix, destination in (("obs__", replay._observations),
                                            ("next_obs__", replay._next_observations)):
                    values = np.asarray(payload[prefix + name], dtype=_storage_dtype(dtype))
                    if values.shape != (used, *shape):
                        raise ValueError(f"invalid replay field shape for {name}")
                    if keep:
                        start = int(order[0])
                        first = min(keep, used - start)
                        destination[name][:first] = values[start:start + first]
                        destination[name][first:keep] = values[:keep - first]
                    del values
            if keep:
                replay._actions[:keep] = actions[order]
                replay._rewards[:keep] = rewards[order]
                replay._dones[:keep] = dones[order]
                replay._episode_ends[:keep] = episode_ends[order]
                replay._ticks_advanced[:keep] = ticks_advanced[order]
                replay._next_action_masks[:keep] = masks[order]
            replay.next_transition_id = int(payload["next_transition_id"]) if "next_transition_id" in payload else saved_size
            replay._size = keep
            replay._position = keep % target_capacity
            first_id = replay.next_transition_id - keep
            if "stream_ids" in payload:
                if (any(name not in payload for name in ('transition_ids','next_ids','stream_schema_version'))
                        or int(payload['stream_schema_version']) != 1):
                    raise ValueError('unsupported or incomplete replay stream metadata')
                for name in ("stream_ids", "transition_ids", "next_ids"):
                    data = np.asarray(payload[name])
                    if data.shape != (used,) or data.dtype.kind not in "iu":
                        raise ValueError("invalid replay stream metadata")
                    getattr(replay, "_"+name)[:keep] = data[order]
                if not np.array_equal(replay._transition_ids[:keep], np.arange(first_id, first_id+keep)):
                    raise ValueError("invalid replay transition identities")
            else:
                replay._stream_ids[:keep] = 0
                replay._transition_ids[:keep] = np.arange(first_id, first_id+keep)
                replay._next_ids[:keep] = np.arange(first_id+1, first_id+keep+1)
                if keep: replay._next_ids[keep-1] = -1
            for slot in range(keep):
                ident = int(replay._transition_ids[slot])
                stream = int(replay._stream_ids[slot])
                if stream < 0: raise ValueError("invalid replay stream id")
                previous = replay._slot_for_id(replay._stream_tails.get(stream, -1))
                if previous is not None and replay._next_ids[previous] == ident:
                    replay._previous_ids[slot] = replay._transition_ids[previous]
                replay._stream_tails[stream] = ident
                successor = int(replay._next_ids[slot])
                if successor >= 0:
                    next_slot = replay._slot_for_id(successor)
                    if successor <= ident or (next_slot is not None and replay._stream_ids[next_slot] != stream):
                        raise ValueError("invalid replay stream successor")
            if "replay_scheme" in payload:
                saved_scheme = str(payload["replay_scheme"].item())
                if saved_scheme in ("uniform", "per"):
                    saved_alpha = float(payload["per_alpha"]) if "per_alpha" in payload else 0.6
                    saved_beta = float(payload["per_beta"]) if "per_beta" in payload else 0.4
                    saved_eps = float(payload["per_eps"]) if "per_eps" in payload else 1e-5
                    saved_max_prio = float(payload["per_max_priority"]) if "per_max_priority" in payload else 1.0
                    replay.set_scheme(saved_scheme, alpha=saved_alpha, beta=saved_beta, eps=saved_eps)
                    replay.max_priority = saved_max_prio
                    if saved_scheme == "per" and "per_priorities" in payload and keep > 0 and replay._tree is not None:
                        saved_priorities = np.asarray(payload["per_priorities"])
                        if saved_priorities.shape[0] >= used:
                            replay._tree.set_all(saved_priorities[order], keep)
                            if "per_raw_priorities" in payload:
                                raw_priorities = np.asarray(payload["per_raw_priorities"], dtype=np.float64)
                                if (raw_priorities.shape != (used,) or not np.isfinite(raw_priorities).all()
                                        or (raw_priorities <= 0).any()):
                                    raise ValueError("invalid raw PER priorities")
                                replay._priorities[:keep] = raw_priorities[order]
                            elif saved_alpha > 0:
                                replay._priorities[:keep] = np.power(saved_priorities[order], 1 / saved_alpha)
                            else:
                                # Legacy alpha=0 snapshots contain no recoverable TD history.
                                replay._priorities[:keep] = saved_max_prio
            try:
                replay._rng.bit_generator.state = json.loads(str(payload["rng_state_json"].item()))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                if strict_resume:
                    raise ValueError("strict_resume cannot restore Replay RNG state") from exc
            metadata = {
                "env_step": int(payload["env_step"]),
                "episode": int(payload["episode"]),
                "saved_capacity": saved_capacity,
                "restored_size": replay._size,
                "credit_assignment": saved_credit,
                "supports_n_step": supports_n_step,
            }
            if "replay_scheme" in payload:
                metadata["replay_scheme"] = str(payload["replay_scheme"].item())
            if "reward_settings_json" in payload:
                settings = json.loads(str(payload["reward_settings_json"].item()))
                if (not isinstance(settings, dict) or type(settings.get("version")) is not int
                        or settings["version"] not in (1, 2, 3, 4, 5, 6) or not isinstance(settings.get("coefficients"), dict)
                        or not isinstance(settings.get("scheme"), str)
                        or (settings["version"] == 2 and (
                            settings["scheme"] != "staged" or settings.get("metrics_version") != 1
                            or settings.get("health_basis") != "initial_native_adventure_spawn_plan"))):
                    raise ValueError("invalid replay reward settings")
                if settings["version"] in (3, 4, 5, 6):
                    from .reward import effective_reward_settings
                    from .config import DoubleDQNConfig
                    from .transition_reward import MOWER_BASIS
                    if settings.get("mower_basis") != MOWER_BASIS:
                        raise ValueError("mower reward semantics changed; rebuild replay while retaining model/optimizer state")
                    mode = "episode_budget" if settings["version"] in (3, 5) else "smooth_total"
                    if settings != effective_reward_settings(DoubleDQNConfig(reward_scheme="transition_dense", shaping_limit_mode=mode)) or not dense_context:
                        raise ValueError("invalid transition_dense replay reward settings")
                replay.reward_settings = settings
                metadata["reward_settings"] = settings
            if "discount_gamma" in payload and "gamma_reference_ticks" in payload:
                metadata["discount_gamma"] = float(payload["discount_gamma"])
                metadata["gamma_reference_ticks"] = int(payload["gamma_reference_ticks"])
        return replay, metadata


def remember_selected_action_mask(action_mask: Any = None) -> None:
    """保留历史接口的兼容函数。当前 DQN 架构下回放仅存储 next_action_masks。"""
    del action_mask


__all__ = ["ReplayBatch", "ReplayBuffer", "remember_selected_action_mask"]
