"""Shared training configuration independent of any RL framework."""

from __future__ import annotations

from dataclasses import dataclass, field

from pvz_gym.runtime import (
    RuntimeConfig, TurboConfig, coerce_config, normalize_simulation_mode,
)
from pvz_gym.types import DEFAULT_MAX_EPISODE_TICKS


_DEFAULT_SIMULATION_MODE = "stepped"


def set_default_simulation_mode(value: str) -> None:
    global _DEFAULT_SIMULATION_MODE
    _DEFAULT_SIMULATION_MODE = normalize_simulation_mode(value)


def get_default_simulation_mode() -> str:
    return _DEFAULT_SIMULATION_MODE


@dataclass(frozen=True)
class EnvConfig:
    level_id: int = 4
    playthrough: int = 1
    ticks_per_step: int = 150
    simulation_mode: str = field(default_factory=get_default_simulation_mode)
    max_episode_ticks: int | None = DEFAULT_MAX_EPISODE_TICKS
    seed_selection_mode: str = "native"
    seed_priority: tuple[int, ...] | None = None
    level_pool: tuple[tuple[int, int], ...] | None = None
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    turbo: TurboConfig = field(default_factory=TurboConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime", coerce_config(RuntimeConfig, self.runtime))
        object.__setattr__(self, "turbo", coerce_config(TurboConfig, self.turbo))
        if self.level_pool is not None:
            pool = tuple((int(lvl), int(p)) for lvl, p in self.level_pool)
            object.__setattr__(self, "level_pool", pool)
        if self.seed_selection_mode not in ("native", "adaptive", "adaptive_shuffled", "custom", "fixed", "fixed_shuffled"):
            raise ValueError("invalid seed_selection_mode")
        if self.seed_priority is not None:
            priority = tuple(self.seed_priority)
            if any(type(i) is not int or not 0 <= i < 40 for i in priority) or len(set(priority)) != len(priority):
                raise ValueError("seed_priority must contain unique ordinary plant IDs")
            object.__setattr__(self, "seed_priority", priority)
        if self.max_episode_ticks is not None and (
            type(self.max_episode_ticks) is not int or self.max_episode_ticks <= 0
        ):
            raise ValueError("max_episode_ticks must be a positive integer or None")
        object.__setattr__(self, "simulation_mode", normalize_simulation_mode(self.simulation_mode))


__all__ = [
    "EnvConfig",
    "get_default_simulation_mode",
    "set_default_simulation_mode",
]
