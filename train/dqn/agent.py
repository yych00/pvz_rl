"""PyTorch Double DQN agent with authoritative action-mask handling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from pvz_gym.action_space import ACTION_SPACE_SIZE, COB_BASE, NOOP, SHOVEL_NORMAL_BASE, decode_action
from pvz_gym.plant_names import plant_name_zh
from pvz_gym.observation import PREVIOUS_OBSERVATION_SCHEMA
from train.common import EnvConfig
from .config import DoubleDQNConfig, saved_discount_settings, saved_exploration_weights
from .exploration import (
    ACTION_FAMILY_NAMES, legal_neighbor_actions, q_guided_fraction,
    sample_balanced_legal_action, sample_neighbor_action,
    sample_q_guided_action,
)
from .networks import NETWORK_SPATIAL, SpatialQNetwork
from .replay import ReplayBatch
from .reward_observation import observation_schema, observation_spec
from .checkpoint_contract import declare_checkpoint
from .learner_timing import phase

try:  # Keep core/unit imports usable when the optional trainer dependency is absent.
    import torch
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - exercised only on machines without PyTorch.
    torch = None
    F = None


def _resolve_device(requested: str):
    if torch is None:  # pragma: no cover
        raise RuntimeError("Double DQN requires PyTorch; run scripts/setup.ps1 to install CUDA PyTorch and training dependencies")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    """Write a checkpoint through a same-directory temp file and atomically replace it."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("Double DQN requires PyTorch; run scripts/setup.ps1 to install CUDA PyTorch and training dependencies")
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


class DoubleDQNAgent:
    def __init__(
        self,
        action_dim: int,
        config: DoubleDQNConfig,
        *,
        device: str | None = None,
        network_type: str = NETWORK_SPATIAL,
    ):
        if torch is None:  # pragma: no cover
            raise RuntimeError("Double DQN requires PyTorch; run scripts/setup.ps1 to install CUDA PyTorch and training dependencies")
        self.action_dim = int(action_dim)
        if self.action_dim != ACTION_SPACE_SIZE:
            raise ValueError(f"DoubleDQNAgent requires action_dim={ACTION_SPACE_SIZE}")
        self.observation_schema = observation_schema(config.reward_scheme == "transition_dense", config.boss_projectiles)
        self.observation_spec = observation_spec(config.reward_scheme == "transition_dense", config.boss_projectiles)
        self.config = config
        self.hidden_sizes = tuple(int(size) for size in config.hidden_sizes)
        self.credit_assignment = config.credit_assignment
        if network_type != NETWORK_SPATIAL:
            raise ValueError(f"network type {network_type!r} is unsupported")
        self.network_type = network_type
        self.device = _resolve_device(device or config.device)
        self._rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        network_class = SpatialQNetwork
        network_kwargs = {"reward_context": config.reward_scheme == "transition_dense",
                          "boss_projectiles": True}
        self.online = network_class(action_dim, self.hidden_sizes, **network_kwargs).to(self.device)
        self.online.initialize_zero_q()
        self.target = network_class(action_dim, self.hidden_sizes, **network_kwargs).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.update_count = 0
        self.last_td_errors: np.ndarray | None = None
        self.last_visualizations: list[dict[str, Any]] = []
        self.learner_profiler = None

    def set_reward_config(self, config):
        """Reward history changes the input contract; no migration is supported."""
        enabled = config.reward_scheme == "transition_dense"
        if enabled != self.online.reward_context:
            raise ValueError("transition_dense uses a different observation contract; start a new model")
        if config.boss_projectiles != bool(getattr(self.online, "boss_projectiles", False)):
            raise ValueError("Boss projectile input is required")
        self.config = config

    def epsilon(self, env_step: int) -> float:
        fraction = min(max(int(env_step), 0) / self.config.epsilon_decay_steps, 1.0)
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def _array_to_tensor(self, value: np.ndarray, *, dtype, use_pinned: bool = False, timing=None):
        """Move one NumPy array to the model device with one dtype conversion.

        Replay batches use pinned host memory/non-blocking CUDA copies. Single
        inference observations deliberately skip pinning because allocating many
        tiny pinned tensors every environment step can cost more than it saves.
        """
        with phase(timing, "tensor_prepare", cuda=False):
            array = np.asarray(value)
            if not array.flags.c_contiguous:
                array = np.ascontiguousarray(array)
            tensor = torch.from_numpy(array)
            if tensor.dtype != dtype:
                tensor = tensor.to(dtype=dtype)
        if self.device.type == "cuda":
            with phase(timing, "pin_memory", cuda=False):
                if use_pinned and not tensor.is_pinned():
                    tensor = tensor.pin_memory()
            with phase(timing, "device_transfer", cuda=False):
                return tensor.to(self.device, non_blocking=use_pinned)
        with phase(timing, "device_transfer", cuda=False):
            return tensor.to(self.device)

    def select_action(
        self,
        observation: Mapping[str, np.ndarray],
        action_mask,
        *,
        env_step: int,
        explore: bool = True,
        visualize: bool = False,
    ) -> int:
        if visualize:
            return self.select_actions([observation], [action_mask], env_step=env_step,
                                       explore=explore, visualize=True)[0]
        self.last_visualizations = []
        mask = np.asarray(action_mask, dtype=np.bool_)
        if mask.shape != (self.action_dim,):
            raise ValueError(f"action mask shape {mask.shape} != {(self.action_dim,)}")
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            raise RuntimeError("PLAYING state has no legal actions")

        exploring = explore and self._rng.random() < self.epsilon(env_step)
        neighbor = False
        if exploring:
            mode = self.config.exploration_mode
            guided = mode != "epsilon_random" and self._rng.random() < q_guided_fraction(
                mode, self.config.exploration_ratio,
            )
            if mode == "epsilon_random" or (mode == "q_guided_mixed" and not guided):
                return sample_balanced_legal_action(
                    mask,
                    self._rng,
                    self.config.exploration_type_weights,
                )
            neighbor = mode == "q_guided_neighbor" and not guided

        with torch.no_grad():
            tensors = self._observation_tensors(observation, batched=False)
            q_values = self.online(tensors)[0]
            torch_mask = self._array_to_tensor(mask, dtype=torch.bool)
            q_values = q_values.masked_fill(~torch_mask, -torch.inf)
            if exploring:
                anchor_mask = mask
                family_weights = self.config.exploration_type_weights
                if neighbor:
                    # Exclude shovel anchors even when all available weights are zero.
                    anchor_mask = mask.copy()
                    anchor_mask[SHOVEL_NORMAL_BASE:COB_BASE] = False
                    family_weights = list(family_weights)
                    family_weights[ACTION_FAMILY_NAMES.index("shovel")] = 0.0
                anchor = sample_q_guided_action(
                    q_values.cpu().numpy(), anchor_mask, self._rng,
                    family_weights,
                )
                if neighbor:
                    neighbors = legal_neighbor_actions(anchor, mask)
                    if neighbors.size:
                        return sample_neighbor_action(
                            neighbors, q_values.cpu().numpy(), self._rng,
                        )
                return anchor
            return int(torch.argmax(q_values).item())

    def select_actions(self, observations, action_masks, *, env_step: int, explore: bool = True,
                       visualize: bool = False) -> list[int]:
        """One network forward for the non-random members of a decision batch."""
        if len(observations) != len(action_masks): raise ValueError("inference batch lengths differ")
        actions = [None] * len(observations)
        self.last_visualizations = []
        q_values_by_index = {}
        forwards, choices = [], {}
        for i, mask in enumerate(action_masks):
            mask = np.asarray(mask, dtype=np.bool_)
            if mask.shape != (self.action_dim,) or not mask.any(): raise ValueError("invalid inference action mask")
            exploring = explore and self._rng.random() < self.epsilon(env_step + i)
            mode = self.config.exploration_mode
            guided = exploring and mode != "epsilon_random" and self._rng.random() < q_guided_fraction(mode, self.config.exploration_ratio)
            if exploring and (mode == "epsilon_random" or (mode == "q_guided_mixed" and not guided)):
                actions[i] = sample_balanced_legal_action(mask, self._rng, self.config.exploration_type_weights)
                if visualize:
                    forwards.append(i)
            else:
                forwards.append(i)
                choices[i] = (mask, exploring, mode == "q_guided_neighbor" and exploring and not guided)
        if forwards:
            batch = {name: np.stack([observations[i][name] for i in forwards]) for name in self.observation_spec}
            with torch.no_grad():
                q_batch = self.online(self._observation_tensors(batch, batched=True)).detach().cpu().numpy()
            for i, values in zip(forwards, q_batch):
                q_values_by_index[i] = values
                if i not in choices:
                    continue
                mask, exploring, neighbor = choices[i]
                values = np.where(mask, values, -np.inf)
                if not exploring:
                    actions[i] = int(np.argmax(values)); continue
                anchor_mask = mask.copy()
                weights = list(self.config.exploration_type_weights)
                if neighbor:
                    anchor_mask[SHOVEL_NORMAL_BASE:COB_BASE] = False
                    weights[ACTION_FAMILY_NAMES.index("shovel")] = 0.0
                anchor = sample_q_guided_action(values, anchor_mask, self._rng, weights)
                neighbors = legal_neighbor_actions(anchor, mask) if neighbor else np.empty(0)
                actions[i] = sample_neighbor_action(neighbors, values, self._rng) if neighbors.size else anchor
        if visualize:
            self.last_visualizations = [
                dict(self.extract_q_visualization_data(obs, mask, raw_q=q_values_by_index[i]),
                     selected_action=int(actions[i]))
                for i, (obs, mask) in enumerate(zip(observations, action_masks))]
        return [int(action) for action in actions]

    def extract_q_visualization_data(
        self,
        observation: Mapping[str, np.ndarray],
        mask: np.ndarray | None = None,
        top_k: int = 5,
        raw_q: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """提取用于热力图展示的 Top-K 非 NOOP 落点动作及 NOOP Q 值。"""
        if raw_q is None:
            with torch.no_grad():
                tensors = self._observation_tensors(observation, batched=False)
                raw_q = self.online(tensors)[0].detach().cpu().numpy()

        noop_q = float(raw_q[NOOP])

        # 候选动作池 (严格排除 NOOP)
        actions = []
        for action_id in range(1, len(raw_q)):
            q_val = float(raw_q[action_id])
            is_legal = bool(mask[action_id]) if mask is not None and len(mask) > action_id else True
            actions.append((q_val, is_legal, action_id))

        # 只展示合法动作；不足 Top-K 时保留实际数量，不用非法动作补足。
        legal_actions = [item for item in actions if item[1]]
        selected = sorted(legal_actions, key=lambda x: x[0], reverse=True)[:top_k]

        top5_list = []
        for q_val, is_legal, action_id in selected:
            decoded = decode_action(action_id)
            desc = ""
            row = -1
            col = -1
            if decoded.type == "plant":
                row = decoded.row
                col = decoded.col
                card_types = observation.get("card_type")
                if card_types is not None and decoded.card_slot < len(card_types):
                    desc = plant_name_zh(int(card_types[decoded.card_slot]) - 1)
                else:
                    desc = f"卡槽{decoded.card_slot + 1}"
            elif decoded.type == "shovel":
                row = decoded.row
                col = decoded.col
                desc = "铲南瓜" if decoded.shovel_target == "pumpkin" else "铲植物"
            elif decoded.type == "cob_fire":
                row = decoded.target_row
                col = decoded.target_col
                desc = "玉米炮发射"
            elif decoded.type == "interact_cell":
                row = decoded.row
                col = decoded.col
                desc = "点击格子"

            top5_list.append({
                "action_id": action_id,
                "q_value": q_val,
                "is_legal": is_legal,
                "row": row,
                "col": col,
                "desc": desc,
            })

        return {
            "top5": top5_list,
            "noop_q": noop_q,
            # Observation categories reserve 0 for padding and shift raw IDs by 1.
            "scene_id": max(0, int(observation["scene"][0]) - 1) if "scene" in observation else 0,
        }

    def _prepare_replay_batch(self, batch: ReplayBatch, *, timing=None):
        """Centralize replay -> torch preparation for one learning update."""
        weights = batch.weights
        if weights is None:
            weights_tensor = torch.ones(len(batch.actions), dtype=torch.float32, device=self.device)
        else:
            weights_tensor = self._array_to_tensor(weights, dtype=torch.float32, use_pinned=True, timing=timing)
        return (
            self._observation_tensors(batch.observations, batched=True, timing=timing),
            self._array_to_tensor(batch.actions, dtype=torch.long, use_pinned=True, timing=timing),
            self._array_to_tensor(batch.rewards, dtype=torch.float32, use_pinned=True, timing=timing),
            self._observation_tensors(batch.next_observations, batched=True, timing=timing),
            self._array_to_tensor(batch.dones, dtype=torch.float32, use_pinned=True, timing=timing),
            self._array_to_tensor(batch.next_action_masks, dtype=torch.bool, use_pinned=True, timing=timing),
            weights_tensor,
        )

    def _transition_discounts(self, batch: ReplayBatch, *, timing=None):
        # Compute in float64 before casting: per-tick gamma is too close to one
        # to exponentiate accurately in float32 for long game-time intervals.
        elapsed = np.asarray(batch.ticks_advanced, dtype=np.float64)
        if elapsed.shape != batch.rewards.shape or not np.isfinite(elapsed).all() or (elapsed < 0).any():
            raise ValueError("invalid batch tick durations")
        discounts = np.power(self.config.gamma, elapsed / self.config.gamma_reference_ticks)
        return self._array_to_tensor(discounts, dtype=torch.float32, timing=timing)

    def _one_step_loss(self, batch: ReplayBatch, *, return_td_errors: bool = False, timing=None):
        selected = self.config.selected_action_forward and hasattr(self.online, "forward_selected_actions")
        legal_next = False
        with phase(timing, "data_prepare"):
            (
                observations,
                actions,
                rewards,
                next_observations,
                dones,
                next_masks,
                weights,
            ) = self._prepare_replay_batch(batch, timing=timing)
            if selected:
                # Pack on CPU before scoring: GPU nonzero/variable-size reads
                # would introduce a synchronization just to discover K.
                with phase(timing, "legal_action_pack", cuda=False):
                    masks = np.asarray(batch.next_action_masks, dtype=np.bool_)
                    counts = masks.sum(axis=1)
                    width = max(1, int(counts.max()))
                    legal_next = width < self.action_dim
                    # If padding would score all actions anyway, use the dense
                    # path and avoid building/transferring candidate tensors.
                    if legal_next:
                        rows, ids = np.nonzero(masks)  # ascending IDs preserve argmax ties
                        candidates = np.zeros((len(counts), width), dtype=np.int64)
                        valid = np.zeros_like(candidates, dtype=np.bool_)
                        columns = np.arange(len(ids)) - np.repeat(np.cumsum(counts) - counts, counts)
                        candidates[rows, columns] = ids
                        valid[rows, columns] = True
                        valid[counts == 0, 0] = True  # preserve existing NOOP fallback
                if legal_next:
                    candidate_ids = self._array_to_tensor(candidates, dtype=torch.long, use_pinned=True, timing=timing)
                    candidate_valid = self._array_to_tensor(valid, dtype=torch.bool, use_pinned=True, timing=timing)

        with phase(timing, "current_forward"):
            q_values = (self.online.forward_selected_actions(observations, actions) if selected else
                        self.online(observations).gather(1, actions.unsqueeze(1)).squeeze(1))

        with torch.no_grad():
            with phase(timing, "online_next_forward"):
                online_next_q = (self.online.forward_selected_actions(next_observations, candidate_ids)
                                 if legal_next else self.online(next_observations))
            with phase(timing, "next_action"):
                if legal_next:
                    online_next_q = online_next_q.masked_fill(~candidate_valid, -torch.inf)
                    next_actions = candidate_ids.gather(1, online_next_q.argmax(dim=1, keepdim=True)).squeeze(1)
                else:
                    safe_masks = next_masks.clone()
                    safe_masks[:, 0] |= ~safe_masks.any(dim=1)
                    online_next_q = online_next_q.masked_fill(~safe_masks, -torch.inf)
                    next_actions = online_next_q.argmax(dim=1)

            with phase(timing, "target_forward"):
                target_next_q = (self.target.forward_selected_actions(next_observations, next_actions) if selected else
                                 self.target(next_observations).gather(1, next_actions.unsqueeze(1)).squeeze(1))
            with phase(timing, "target_compute"):
                targets = rewards + self._transition_discounts(batch, timing=timing) * (1.0 - dones) * target_next_q

        with phase(timing, "loss_compute"):
            elementwise_loss = F.smooth_l1_loss(q_values, targets, reduction="none")
            weighted_loss = torch.mean(elementwise_loss * weights)
            if return_td_errors:
                td_errors = (q_values - targets).detach().abs()
                return weighted_loss, td_errors
        return weighted_loss

    def learn(self, batch: ReplayBatch, *, return_td_errors: bool = False) -> float | tuple[float, np.ndarray]:
        profiling = nullcontext(None)
        if self.learner_profiler is not None:
            profiling = self.learner_profiler.update(
                self.update_count,
                force_cuda=(self.update_count + 1) % self.config.target_update_interval == 0,
            )
        with profiling as timing:
            loss, td_errors = self._one_step_loss(batch, return_td_errors=True, timing=timing)
            with phase(timing, "zero_grad"):
                self.optimizer.zero_grad(set_to_none=True)
            with phase(timing, "backward"):
                loss.backward()
            with phase(timing, "grad_clip"):
                torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.config.max_grad_norm)
            with phase(timing, "optimizer"):
                self.optimizer.step()

            self.update_count += 1
            if self.update_count % self.config.target_update_interval == 0:
                with phase(timing, "target_sync"):
                    self.target.load_state_dict(self.online.state_dict())
            with phase(timing, "result_copy"):
                loss_val = float(loss.detach().cpu().item())
                td_errors_np = td_errors.detach().cpu().numpy()
                self.last_td_errors = td_errors_np
        if return_td_errors:
            return loss_val, td_errors_np
        return loss_val

    def _observation_tensors(self, observation, *, batched: bool, timing=None):
        if not isinstance(observation, Mapping) or set(observation) != set(self.observation_spec):
            raise ValueError("observation does not match the structured Observation contract")
        result = {}
        for name, (shape, dtype) in self.observation_spec.items():
            value = np.asarray(observation[name])
            expected = (None, *shape) if batched else shape
            if (batched and (value.ndim != len(shape) + 1 or value.shape[1:] != shape)) or (
                not batched and value.shape != shape
            ):
                raise ValueError(f"observation field {name} shape {value.shape} != {expected}")
            torch_dtype = torch.float32 if np.issubdtype(dtype, np.floating) else torch.long
            tensor = self._array_to_tensor(value, dtype=torch_dtype, use_pinned=batched, timing=timing)
            result[name] = tensor if batched else tensor.unsqueeze(0)
        return result

    def checkpoint_payload(
        self,
        *,
        env_step: int,
        episode: int,
        training_summary: dict[str, Any] | None = None,
        episode_rng_state: dict[str, Any] | None = None,
        seed_types: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        config_payload = asdict(self.config)
        # The controller may override runtime schedules/rewards after loading a
        # checkpoint, but the instantiated network shape cannot change. Persist
        # the actual architecture so the next load always reconstructs a network
        # compatible with the saved state_dict.
        config_payload["hidden_sizes"] = self.hidden_sizes
        config_payload["credit_assignment"] = self.credit_assignment
        config_payload["boss_projectiles"] = bool(getattr(self.online, "boss_projectiles", False))
        payload = {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "observation_schema": self.observation_schema,
            "action_dim": self.action_dim,
            "network_type": self.network_type,
            "training_device": str(self.device),
            "config": config_payload,
            "env_step": int(env_step),
            "episode": int(episode),
            "update_count": int(self.update_count),
            "training_summary": dict(training_summary or {}),
            "seed_types": list(seed_types) if seed_types else None,
            "agent_rng_state": self._rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
        }
        if hasattr(self, "sampling_state"):
            payload["sampling_state"] = self.sampling_state
        if hasattr(self, "curriculum_state"):
            payload["curriculum_state"] = self.curriculum_state
        if self.device.type == "cuda" and torch.cuda.is_available():
            payload["cuda_rng_states"] = torch.cuda.get_rng_state_all()
        if episode_rng_state is not None:
            payload["episode_rng_state"] = episode_rng_state
        return declare_checkpoint(payload)

    def save(self, path: str | Path, *, env_step: int, episode: int,
             training_summary=None, episode_rng_state=None, seed_types=None) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _atomic_torch_save(self.checkpoint_payload(env_step=env_step, episode=episode,
            training_summary=training_summary, episode_rng_state=episode_rng_state,
            seed_types=seed_types), destination)
        return destination

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str = "auto",
        weights_only: bool = False,
        boss_projectiles: bool | None = None,
        strict_resume: bool = False,
    ) -> tuple["DoubleDQNAgent", DoubleDQNConfig, dict[str, Any]]:
        if torch is None:  # pragma: no cover
            raise RuntimeError("Double DQN requires PyTorch; run scripts/setup.ps1 to install CUDA PyTorch and training dependencies")
        map_location = _resolve_device(device)
        try:
            payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        except TypeError:  # Older PyTorch releases do not expose weights_only.
            payload = torch.load(Path(path), map_location=map_location)

        if strict_resume:
            from .checkpoint_contract import inspect_checkpoint
            if weights_only:
                raise ValueError("strict_resume cannot be combined with weights_only")
            if inspect_checkpoint(payload) != "full":
                raise ValueError("strict_resume requires a full checkpoint; model-only or legacy contents cannot guarantee restoration")
            required = ("target", "optimizer", "update_count", "agent_rng_state",
                        "torch_rng_state", "episode_rng_state")
            missing = [key for key in required if payload.get(key) is None]
            if missing:
                raise ValueError("strict_resume is missing training state: " + ", ".join(missing))
            if map_location.type == "cuda" and len(payload.get("cuda_rng_states", [])) != torch.cuda.device_count():
                raise ValueError("strict_resume requires saved CUDA RNG state for every current device")
            if not isinstance(payload["optimizer"], dict) or not {"state", "param_groups"}.issubset(payload["optimizer"]):
                raise ValueError("strict_resume has invalid optimizer state")

        config_data = dict(payload["config"])
        config_data.setdefault("evaluation_mode", "random")
        config_data.setdefault("shaping_limit_mode", "episode_budget")
        if config_data.get("boss_projectiles") is not True:
            raise ValueError("Boss projectile input is required; legacy models without it are unsupported")
        config_data.setdefault("n_step", 1)
        config_data.setdefault("replay_scheme", "uniform")
        env_data = dict(config_data.pop("env"))
        source_ticks = config_data.get("gamma_reference_ticks", env_data.get("ticks_per_step", 150))
        normalized = saved_discount_settings(config_data, env_data.get("ticks_per_step", 150))
        if source_ticks != normalized["gamma_reference_ticks"]:
            payload["gamma_conversion"] = {
                "source_gamma": config_data.get("gamma", 0.9995), "source_ticks": source_ticks,
                "gamma": normalized["gamma"], "reference_ticks": normalized["gamma_reference_ticks"],
            }
        config_data.update(normalized)
        if config_data.get("exploration_type_weights") is not None:
            config_data["exploration_type_weights"] = saved_exploration_weights(config_data["exploration_type_weights"])
        hidden_sizes = config_data.get("hidden_sizes")
        if hidden_sizes is not None:
            config_data["hidden_sizes"] = tuple(hidden_sizes)
        expected_schema = observation_schema(config_data.get("reward_scheme", DoubleDQNConfig().reward_scheme) == "transition_dense",
                                              config_data["boss_projectiles"])
        schema = str(payload.get("observation_schema", ""))
        if schema != expected_schema:
            previous_schema = PREVIOUS_OBSERVATION_SCHEMA + (
                "_transition_dense_context_v1" if config_data.get("reward_scheme") == "transition_dense" else ""
            )
            if schema != previous_schema or strict_resume:
                raise ValueError(
                    f"checkpoint observation schema {schema!r} is not {expected_schema!r}"
                )
            # Tensor/parameter shapes are unchanged. Keep weights and ordinary
            # resume state, but never restore Replay with the old spatial meaning.
            payload["observation_migration"] = {
                "source": schema, "destination": expected_schema,
                "reason": "Boss body is nonspatial; rebuild Replay",
            }
        network_type = str(payload.get("network_type", ""))
        if network_type != NETWORK_SPATIAL:
            raise ValueError(f"checkpoint network type {network_type!r} is unsupported")
        config_data.setdefault("credit_assignment", "one_step")
        config_data.setdefault("training_architecture", "legacy")
        config = DoubleDQNConfig(env=EnvConfig(**env_data), **config_data)
        if boss_projectiles is not None and boss_projectiles is not True:
            raise ValueError("Boss projectile input is required")
        agent = cls(
            int(payload["action_dim"]),
            config,
            device=str(map_location),
            network_type=NETWORK_SPATIAL if weights_only else network_type,
        )
        agent.online.load_state_dict(payload["online"])
        if weights_only:
            agent.target.load_state_dict(agent.online.state_dict())
            agent.target.eval()
            return agent, config, payload
        agent.target.load_state_dict(payload.get("target", payload["online"]))
        optimizer_state = payload.get("optimizer")
        if optimizer_state is not None:
            try:
                agent.optimizer.load_state_dict(optimizer_state)
                for state in agent.optimizer.state.values():
                    for key, value in tuple(state.items()):
                        if torch.is_tensor(value):
                            state[key] = value.to(agent.device)
            except Exception as exc:
                if strict_resume:
                    raise ValueError("strict_resume cannot restore optimizer state") from exc
                agent.optimizer = torch.optim.Adam(agent.online.parameters(), lr=config.learning_rate)
        agent.update_count = int(payload.get("update_count", 0))
        if "sampling_state" in payload:
            agent.sampling_state = payload["sampling_state"]
        if "agent_rng_state" in payload:
            agent._rng.bit_generator.state = payload["agent_rng_state"]
        elif int(payload.get("env_step", 0)) > 0:
            # Historical draws cannot be reconstructed. Use a progress-specific
            # stream instead of repeating the beginning of the original seed.
            agent._rng = np.random.default_rng(np.random.SeedSequence([
                config.seed, int(payload["env_step"]), int(payload.get("episode", 0)), 0,
            ]))
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"].cpu())
        if torch.cuda.is_available():
            for index, state in enumerate(payload.get("cuda_rng_states", [])[:torch.cuda.device_count()]):
                torch.cuda.set_rng_state(state.cpu(), device=index)
        agent.online.train()
        agent.target.eval()
        return agent, config, payload


__all__ = ["DoubleDQNAgent"]
