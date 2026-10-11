"""Utilities for resuming a Double DQN run directory."""

from __future__ import annotations

import json
import pickle
from dataclasses import fields
from pathlib import Path
from typing import Any

from train.common import EnvConfig

from .config import DoubleDQNConfig, saved_discount_settings, saved_exploration_weights


def _filtered_dataclass_kwargs(cls, values: dict[str, Any]) -> dict[str, Any]:
    allowed = {item.name for item in fields(cls)}
    return {key: value for key, value in values.items() if key in allowed}


def _load_payload(run_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    directory = Path(run_dir)
    config_path = directory / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"run config not found: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid run config: {config_path}")
    return directory, payload


def load_run_config(run_dir: str | Path) -> DoubleDQNConfig:
    """Load a saved run config while tolerating older/newer config keys."""
    _directory, payload = _load_payload(run_dir)
    config_data = dict(payload)
    config_data.setdefault("shaping_limit_mode", "episode_budget")
    config_data.setdefault("training_architecture", "legacy")
    config_data.setdefault("n_step", 1)
    config_data.setdefault("evaluation_mode", "random")
    config_data.setdefault("replay_scheme", "uniform")
    if config_data.get("boss_projectiles") is not True:
        raise ValueError("Boss projectile input is required; legacy models without it are unsupported")
    if "event_credit_weight" in config_data:
        raise ValueError("event_credit_weight is unsupported")
    # Only ordinary TD is supported; preserve the marker when explicitly saved.
    config_data.setdefault("credit_assignment", "one_step")
    env_data = config_data.pop("env", {})
    if not isinstance(env_data, dict):
        raise ValueError(f"invalid env config in: {Path(run_dir) / 'config.json'}")
    hidden_sizes = config_data.get("hidden_sizes")
    if hidden_sizes is not None:
        config_data["hidden_sizes"] = tuple(int(value) for value in hidden_sizes)
    type_weights = config_data.get("exploration_type_weights")
    if type_weights is not None:
        config_data["exploration_type_weights"] = saved_exploration_weights(type_weights)

    env = EnvConfig(**_filtered_dataclass_kwargs(EnvConfig, env_data))
    config_data.update(saved_discount_settings(config_data, env.ticks_per_step))
    return DoubleDQNConfig(
        env=env,
        **_filtered_dataclass_kwargs(DoubleDQNConfig, config_data),
    )


def load_run_seed_types(run_dir: str | Path) -> tuple[int, ...] | None:
    """Restore the optional fixed card deck recorded for the run."""
    _directory, payload = _load_payload(run_dir)
    raw = payload.get("seed_types")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("seed_types in run config must be a list or null")
    values = tuple(int(value) for value in raw)
    return values or None


def load_checkpoint_seed_types(path: str | Path, config: DoubleDQNConfig,
                               payload: dict[str, Any]) -> tuple[int, ...] | None:
    """New checkpoints carry their deck; old ones may use a matching run sidecar."""
    if "seed_types" in payload:
        raw = payload["seed_types"]
        return tuple(int(value) for value in raw) if raw else None
    directory = Path(path).parent
    try:
        sibling = load_run_config(directory)
        if (sibling.env.level_id, sibling.env.playthrough) == (
            config.env.level_id, config.env.playthrough,
        ):
            return load_run_seed_types(directory)
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None


def load_episode_summary(
    run_dir: str | Path,
    *,
    up_to_episode: int | None = None,
) -> dict[str, Any]:
    """Summarize episodes.jsonl for old checkpoints without embedded metrics."""
    metrics_path = Path(run_dir) / "episodes.jsonl"
    if not metrics_path.is_file():
        return {}

    records: dict[int, dict[str, Any]] = {}
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            episode = int(record.get("episode", 0) or 0)
            if episode <= 0 or (up_to_episode is not None and episode > up_to_episode):
                continue
            records[episode] = record

    ordered = [records[key] for key in sorted(records)]
    if not ordered:
        return {}
    wins = sum(str(item.get("result", "UNKNOWN")) == "WON" for item in ordered)
    losses = sum(str(item.get("result", "UNKNOWN")) == "LOST" for item in ordered)
    terminal = wins + losses
    training_rewards = [float(item.get("training_episode_reward", 0.0) or 0.0) for item in ordered]
    lengths = [max(0, int(item.get("episode_steps", 0) or 0)) for item in ordered]
    last = ordered[-1]
    return {
        "env_steps": int(last.get("env_step", 0) or 0),
        "episodes": int(last.get("episode", len(ordered)) or len(ordered)),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / terminal if terminal else 0.0,
        "invalid_actions": sum(max(0, int(item.get("invalid_actions", 0) or 0)) for item in ordered),
        "mowers_used": sum(max(0, int(item.get("mowers_used", 0) or 0)) for item in ordered),
        "last_loss": last.get("loss"),
        "epsilon": float(last.get("epsilon", 0.0) or 0.0),
        "mean_training_reward": sum(training_rewards) / len(training_rewards),
        "mean_episode_length": sum(lengths) / len(lengths),
        "tick_samples": int(last.get("tick_samples", 0) or 0),
        "mean_ticks_advanced": float(last.get("mean_ticks_advanced", 0.0) or 0.0),
        "std_ticks_advanced": float(last.get("std_ticks_advanced", 0.0) or 0.0),
        "long_tick_jumps": int(last.get("long_tick_jumps", 0) or 0),
        "long_tick_jump_rate": float(last.get("long_tick_jump_rate", 0.0) or 0.0),
    }


def resolve_resume_checkpoint(run_dir: str | Path, *, mode: str = "resume") -> Path:
    """Pick the newest safe checkpoint in an existing run directory.

    Rolling latest, stopped/final snapshots and milestones are all considered.
    Modification time is authoritative so an old stopped.pt cannot override a
    newer crash-recovery latest.pt after the run has already continued.
    """
    directory = Path(run_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"run directory not found: {directory}")

    named = [directory / name for name in ("stopped.pt", "latest.pt", "latest-full.pt", "latest-model.pt", "final.pt")]
    milestones = list(directory.glob("checkpoint-*.pt"))
    candidates = [path for path in (*named, *milestones) if path.is_file()]
    if mode in ("strict_resume", "weights_replay"):
        import numpy as np
        import torch
        from .checkpoint_contract import inspect_checkpoint
        valid = []
        for path in candidates:
            try:
                model = torch.load(path, map_location="cpu", weights_only=False)
                kind = inspect_checkpoint(model)
                if kind == "model_only" or (mode == "strict_resume" and kind != "full"):
                    continue
                if mode == "strict_resume" and any(model.get(key) is None for key in (
                        "target", "optimizer", "update_count", "agent_rng_state", "torch_rng_state", "episode_rng_state")):
                    continue
                name = model.get("replay_file", "replay.npz")
                if Path(name).name != name:
                    continue
                with np.load(directory / name, allow_pickle=False) as replay:
                    if (int(replay["env_step"]) != int(model["env_step"])
                            or int(replay["episode"]) != int(model["episode"])
                            or str(replay["observation_schema"].item()) != model["observation_schema"]):
                        continue
                    if mode == "weights_replay":
                        valid.append(path)
                        continue
                    required = ("rng_state_json", "stream_ids", "transition_ids", "next_ids",
                                "stream_schema_version", "reward_settings_json", "discount_gamma",
                                "gamma_reference_ticks", "raw_return_schema_version", "supports_n_step", "episode_ends")
                    if any(key not in replay for key in required):
                        continue
                    if str(replay.get("replay_scheme", np.asarray("uniform")).item()) == "per" and any(
                            key not in replay for key in ("per_alpha", "per_beta", "per_eps", "per_max_priority",
                                                          "per_priorities", "per_raw_priorities")):
                        continue
                    rng = np.random.default_rng()
                    rng.bit_generator.state = json.loads(str(replay["rng_state_json"].item()))
                valid.append(path)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError):
                continue
        candidates = valid
        if not candidates:
            raise ValueError(f"{mode} found no full checkpoint with a matching Replay in: {directory}")
    if not candidates:
        raise FileNotFoundError(f"no Double DQN checkpoint found in: {directory}")

    tie_priority = {"stopped.pt": 4, "latest-full.pt": 3, "latest.pt": 2, "final.pt": 1}
    return max(
        candidates,
        key=lambda path: (
            path.stat().st_mtime_ns,
            tie_priority.get(path.name, 0),
            path.name,
        ),
    )


__all__ = [
    "load_episode_summary",
    "load_run_config",
    "load_run_seed_types",
    "load_checkpoint_seed_types",
    "resolve_resume_checkpoint",
]
