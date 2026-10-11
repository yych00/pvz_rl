"""Hyperparameters for the first model-free value baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from train.common import EnvConfig
from train.runtime import TrainingRuntimeConfig
from pvz_gym.runtime import coerce_config
from .reward import REWARD_SCHEMES
from .exploration import DEFAULT_EXPLORATION_MODE, DEFAULT_EXPLORATION_RATIO, q_guided_fraction

CREDIT_ASSIGNMENT_MODES = ("one_step",)
REPLAY_SCHEMES = ("uniform", "per")
GAMMA_REFERENCE_TICKS = 100
DISCOUNT_SCHEMA_VERSION = 1
DEFAULT_EXPLORATION_TYPE_WEIGHTS: tuple[float, float, float, float, float] = (
    2.0, 4.0, 0.1, 0.1, 0.1
)


def convert_gamma(gamma: float, source_ticks: int, target_ticks: int = GAMMA_REFERENCE_TICKS) -> float:
    """Change the time unit without changing the underlying discount rate."""
    if not math.isfinite(gamma) or not 0 < gamma <= 1:
        raise ValueError("gamma must be finite and in (0, 1]")
    if any(type(ticks) is not int or ticks <= 0 for ticks in (source_ticks, target_ticks)):
        raise ValueError("gamma reference ticks must be positive integers")
    if source_ticks == target_ticks:
        return float(gamma)
    converted = math.exp(math.log(gamma) * target_ticks / source_ticks)
    if converted == 0:
        raise ValueError("converted gamma is too small to represent")
    return converted


def saved_discount_settings(values: dict, legacy_ticks_per_step: int = 150) -> dict:
    """Normalize step-based and arbitrary tick-based snapshots to 100 ticks.

    Missing version identifies existing snapshots; an unknown future version
    is rejected rather than guessing the meaning of its discount settings.
    """
    # Temporary compatibility entry point. Retire source-unit conversion in a
    # later cleanup after legacy snapshots are migrated; new snapshots use 100.
    version = values.get("discount_schema_version", DISCOUNT_SCHEMA_VERSION)
    if type(version) is not int or version != DISCOUNT_SCHEMA_VERSION:
        raise ValueError(f"unsupported discount_schema_version: {version!r}")
    if "discount_schema_version" in values and "gamma_reference_ticks" not in values:
        raise ValueError("versioned discount config is missing gamma_reference_ticks")
    reference = values.get("gamma_reference_ticks", legacy_ticks_per_step)
    gamma = values.get("gamma", 0.9995)  # Historical default for incomplete old configs.
    return {
        "gamma": convert_gamma(gamma, reference),
        "gamma_reference_ticks": GAMMA_REFERENCE_TICKS,
        "discount_schema_version": DISCOUNT_SCHEMA_VERSION,
    }


def saved_exploration_weights(values) -> tuple[float, ...]:
    """Old four-family snapshots predate INTERACT; keep its exploration off."""
    weights = tuple(float(value) for value in values)
    return weights + (0.0,) if len(weights) == 4 else weights


def _default_env_config() -> EnvConfig:
    """Use the fast deterministic simulation path for DQN training by default.

    Other trainers keep their own defaults, while callers can still explicitly
    request ``stepped`` or ``continuous`` when debugging or comparing behavior.
    """
    return EnvConfig(simulation_mode="headless")


def validate_resume_config(requested: DoubleDQNConfig, saved: DoubleDQNConfig) -> None:
    """A continued run keeps its architecture, task, and random seed."""
    changed = [key for key in ("hidden_sizes", "credit_assignment", "seed", "boss_projectiles")
               if getattr(requested, key) != getattr(saved, key)]
    changed.extend(f"env.{key}" for key in ("level_id", "playthrough", "level_pool")
                   if getattr(requested.env, key) != getattr(saved.env, key))
    if changed:
        raise ValueError("接续训练不能修改 " + ", ".join(changed)
                         + "；切换关卡或种子请使用权重微调模式。")


@dataclass(frozen=True)
class DoubleDQNConfig:
    env: EnvConfig = field(default_factory=_default_env_config)
    total_steps: int = 250_000
    replay_capacity: int = 200_000
    learning_starts: int = 10_000
    batch_size: int = 128
    n_step: int = 3
    gamma: float = 0.999
    # 100 native ticks = one second of normal game time. This unit stays fixed
    # when the policy's decision interval or simulation speed changes.
    gamma_reference_ticks: int = GAMMA_REFERENCE_TICKS
    discount_schema_version: int = DISCOUNT_SCHEMA_VERSION
    learning_rate: float = 1e-4
    train_every: int = 6
    auto_train_every: bool = True
    auto_train_high_watermark: int = 16
    auto_train_low_watermark: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 2_000
    epsilon_start: float = 0.75
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 200_000
    exploration_mode: str = DEFAULT_EXPLORATION_MODE
    exploration_ratio: tuple[float, float] = DEFAULT_EXPLORATION_RATIO
    # NOOP, PLANT, SHOVEL, COB, INTERACT: family selection for random/Q guidance.
    # Neighbor anchors exclude shovel; greedy uses masked argmax.
    exploration_type_weights: tuple[float, float, float, float, float] = (
        DEFAULT_EXPLORATION_TYPE_WEIGHTS
    )
    hidden_sizes: tuple[int, int] = (512, 256)
    reward_scheme: str = "transition_dense"
    # New runs smoothly bound the net episode shaping return to +/-0.40.
    # episode_budget is retained for evaluating/resuming historical checkpoints.
    shaping_limit_mode: str = "smooth_total"
    credit_assignment: str = "one_step"
    replay_scheme: str = "per"
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_steps: int = 100_000
    per_eps: float = 1e-5
    # Legacy sparse-scheme scalar overrides kept for old checkpoints/CLI calls.
    reward_scale: float = 0.01
    mower_penalty: float = 0.1
    max_grad_norm: float = 10.0
    # Lightweight model-only autosaves stay frequent; full checkpoints (which
    # include replay.npz) are intentionally less frequent because compressed
    # replay snapshots can stall the training process for a long time.
    autosave_every: int = 10_000
    checkpoint_every: int = 50_000
    seed: int = 0
    device: str = "auto"
    selected_action_forward: bool = True
    boss_projectiles: bool = True
    training_architecture: str = "actor_learner"
    policy_sync_interval: int = 32
    max_pending_updates: int = 128
    evaluation_mode: str | None = None
    evaluation_per_level: int = 4
    evaluation_random_episodes: int = 20
    level_sampling_mode: str = "squared_failure"
    adaptive_level_sampling: bool = True
    level_sampling_window: int = 40
    level_sampling_update_every: int = 20
    level_sampling_uniform_mix: float = 0.05
    level_sampling_max_weight: float = 4.0
    runtime: TrainingRuntimeConfig = field(default_factory=TrainingRuntimeConfig)

    def __post_init__(self) -> None:
        if self.evaluation_mode is None:
            is_multi = bool(self.env.level_pool and len(self.env.level_pool) > 1)
            object.__setattr__(self, "evaluation_mode", "per_level" if is_multi else "random")
        elif self.evaluation_mode.lower() in ("off", "none"):
            object.__setattr__(self, "evaluation_mode", "off")
        elif self.evaluation_mode not in ("per_level", "random"):
            raise ValueError("evaluation_mode must be per_level, random, or off")
        for name in ("evaluation_per_level", "evaluation_random_episodes",
                     "level_sampling_window", "level_sampling_update_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.level_sampling_mode not in ("bounded_difficulty", "squared_failure", "uniform"):
            raise ValueError("invalid level_sampling_mode")
        if type(self.adaptive_level_sampling) is not bool:
            raise ValueError("adaptive_level_sampling must be a boolean")
        if not math.isfinite(self.level_sampling_uniform_mix) or not 0 <= self.level_sampling_uniform_mix <= 1:
            raise ValueError("level_sampling_uniform_mix must be in [0, 1]")
        if not math.isfinite(self.level_sampling_max_weight) or self.level_sampling_max_weight < 1:
            raise ValueError("level_sampling_max_weight must be finite and >= 1")
        if self.shaping_limit_mode not in ("smooth_total", "episode_budget"):
            raise ValueError("shaping_limit_mode must be smooth_total or episode_budget")
        if type(self.selected_action_forward) is not bool or type(self.boss_projectiles) is not bool:
            raise ValueError("selected_action_forward and boss_projectiles must be booleans")
        if not self.boss_projectiles:
            raise ValueError("Boss projectile input is required; legacy models without it are unsupported")
        object.__setattr__(self, "env", coerce_config(EnvConfig, self.env))
        object.__setattr__(self, "runtime", coerce_config(TrainingRuntimeConfig, self.runtime))
        if self.training_architecture not in ("legacy", "actor_learner"):
            raise ValueError("training_architecture must be legacy or actor_learner")
        for name in ("policy_sync_interval", "max_pending_updates"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.training_architecture == "actor_learner" and self.runtime.max_policy_lag < 1:
            raise ValueError("actor_learner requires max_policy_lag >= 1")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.replay_capacity <= 0:
            raise ValueError("replay_capacity must be positive")
        if self.learning_starts < 0:
            raise ValueError("learning_starts must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if type(self.n_step) is not int or not 1 <= self.n_step <= self.replay_capacity:
            raise ValueError("n_step must be an integer in [1, replay_capacity]")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be >= batch_size")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if type(self.gamma_reference_ticks) is not int or self.gamma_reference_ticks <= 0:
            raise ValueError("gamma_reference_ticks must be a positive integer")
        if type(self.discount_schema_version) is not int or self.discount_schema_version != DISCOUNT_SCHEMA_VERSION:
            raise ValueError("unsupported discount_schema_version")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.train_every <= 0 or self.gradient_steps <= 0:
            raise ValueError("train cadence must be positive")
        if type(self.auto_train_every) is not bool:
            raise ValueError("auto_train_every must be a boolean")
        for name in ("auto_train_high_watermark", "auto_train_low_watermark"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.auto_train_low_watermark >= self.auto_train_high_watermark:
            raise ValueError("auto_train_low_watermark must be strictly less than auto_train_high_watermark")
        if self.target_update_interval <= 0:
            raise ValueError("target_update_interval must be positive")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon schedule must satisfy 0 <= end <= start <= 1")
        if self.epsilon_decay_steps <= 0:
            raise ValueError("epsilon_decay_steps must be positive")
        ratio = tuple(float(value) for value in self.exploration_ratio)
        q_guided_fraction(self.exploration_mode, ratio)
        object.__setattr__(self, "exploration_ratio", ratio)
        weights = tuple(float(value) for value in self.exploration_type_weights)
        if len(weights) != 5:
            raise ValueError(
                "exploration_type_weights must contain noop, plant, shovel, cob, and interact weights"
            )
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("exploration_type_weights must be finite and non-negative")
        if not any(value > 0.0 for value in weights):
            raise ValueError("at least one exploration type weight must be positive")
        object.__setattr__(self, "exploration_type_weights", weights)
        if len(self.hidden_sizes) != 2 or any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain two positive layer sizes")
        if self.reward_scheme not in REWARD_SCHEMES:
            raise ValueError(
                f"reward_scheme must be one of {', '.join(REWARD_SCHEMES)}"
            )
        if self.credit_assignment not in CREDIT_ASSIGNMENT_MODES:
            raise ValueError(f"credit_assignment must be one of {CREDIT_ASSIGNMENT_MODES}")
        if self.replay_scheme not in REPLAY_SCHEMES:
            raise ValueError(f"replay_scheme must be one of {REPLAY_SCHEMES}")
        if not 0.0 <= self.per_alpha <= 1.0:
            raise ValueError("per_alpha must be in [0, 1]")
        if not 0.0 <= self.per_beta_start <= 1.0:
            raise ValueError("per_beta_start must be in [0, 1]")
        if not 0.0 <= self.per_beta_end <= 1.0:
            raise ValueError("per_beta_end must be in [0, 1]")
        if self.per_beta_steps <= 0:
            raise ValueError("per_beta_steps must be positive")
        if self.per_eps <= 0.0:
            raise ValueError("per_eps must be positive")
        if self.reward_scale <= 0.0:
            raise ValueError("reward_scale must be positive")
        if self.mower_penalty < 0.0:
            raise ValueError("mower_penalty must be non-negative")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.autosave_every < 0:
            raise ValueError("autosave_every must be non-negative")
        if self.checkpoint_every < 0:
            raise ValueError("checkpoint_every must be non-negative")


__all__ = [
    "CREDIT_ASSIGNMENT_MODES",
    "DEFAULT_EXPLORATION_TYPE_WEIGHTS",
    "DoubleDQNConfig",
    "REPLAY_SCHEMES",
    "convert_gamma",
    "saved_discount_settings",
    "saved_exploration_weights",
]
