"""Double-DQN reward schemes and trainer-level shaping.

The core :mod:`pvz_gym` environment remains algorithm-agnostic.  The environment
may expose neutral diagnostics (for example zombie threat delta and terminal
board scores), while this module owns the actual Double-DQN reward weights.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import DoubleDQNConfig


@dataclass(frozen=True)
class RewardScheme:
    name: str
    label: str
    terminal_scale: float
    mower_used: float = 0.0
    plant_death: float = 0.0
    zombie_kill: float = 0.0
    wave_progress: float = 0.0
    sun_collect: float = 0.0
    zombie_threat: float = 0.0
    terminal_plants: float = 0.0
    terminal_sun: float = 0.0
    description: str = ""


REWARD_SCHEMES: dict[str, RewardScheme] = {
    "sparse": RewardScheme(
        name="sparse",
        label="Sparse",
        terminal_scale=0.01,  # native terminal reward is +/-100 -> +/-1
        mower_used=-0.10,
        description="Terminal reward plus mower penalty only.",
    ),
    "balanced": RewardScheme(
        name="balanced",
        label="Balanced",
        terminal_scale=0.01,  # WIN +1 / LOSS -1
        mower_used=-0.20,
        zombie_threat=0.20,
        terminal_plants=0.30,
        terminal_sun=0.10,
        description=(
            "General Double-DQN shaping: win/loss dominates; mower loss is immediate; "
            "threat uses bounded state progress; terminal plants and sun are small tie-breakers."
        ),
    ),
    "dense_early": RewardScheme(
        name="dense_early",
        label="Dense early",
        terminal_scale=0.01,  # WIN +1 / LOSS -1
        mower_used=-0.20,
        plant_death=-0.01,
        zombie_kill=0.003,
        wave_progress=0.03,
        sun_collect=0.0001,
        description="Early-level shaping for survival, kills, wave progress and economy.",
    ),
    "transition_dense": RewardScheme(
        name="transition_dense", label="Transition dense", terminal_scale=0.01,
        mower_used=-0.10,
        description="Per-transition assets and conservative damage; first mower -0.50, later -0.10; net episode shaping smoothly approaches +/-0.40 with positive/negative cancellation; matched-cohort threat x0.10 clipped +/-0.01.",
    ),
    "staged": RewardScheme(
        name="staged", label="Staged", terminal_scale=0.01, mower_used=-0.20,
        description=("Original-price inventory and planned-health damage progress; "
                     "+0.10 auxiliary cap per 1000 ticks (max +1.00 total), mower -0.20, "
                     "terminal win/loss plus 0.50 damage fraction."),
    ),
}
REWARD_SCHEME_NAMES = tuple(REWARD_SCHEMES)


def get_reward_scheme(name: str) -> RewardScheme:
    try:
        return REWARD_SCHEMES[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"unknown reward scheme {name!r}; expected one of {', '.join(REWARD_SCHEME_NAMES)}"
        ) from exc


def effective_reward_settings(config: "DoubleDQNConfig") -> dict[str, Any]:
    """Identify stored reward semantics, excluding discount and loss schedules."""
    scheme = get_reward_scheme(config.reward_scheme)
    if scheme.name == "transition_dense":
        from . import transition_reward as dense
        smooth_total = config.shaping_limit_mode == "smooth_total"
        settings = {"version": 6 if smooth_total else 5, "scheme": scheme.name, "metrics_version": dense.METRICS_VERSION,
                "health_basis": "initial_native_adventure_spawn_plan",
                "mower_basis": dense.MOWER_BASIS,
                "boss_basis": dense.BOSS_BASIS,
                "asset_basis": "original_price_consumable_inventory_and_plants",
                "damage_basis": "conservative_snapshot_low_watermark_mower_rows_excluded",
                "observation_context": "first_mower_and_net_balance_headroom_v1" if smooth_total else "first_mower_and_remaining_budgets_v1",
                "threat_basis": "matched_hostile_snapshot_cohort_mower_rows_excluded",
                "coefficients": {"asset_reference": dense.ASSET_REFERENCE,
                    "damage": dense.DAMAGE_WEIGHT, "threat": dense.THREAT_WEIGHT,
                    "threat_spawn_x": dense.THREAT_SPAWN_X, "threat_house_x": dense.THREAT_HOUSE_X,
                    "threat_step_cap": dense.THREAT_STEP_CAP,
                    "positive_cap": dense.POSITIVE_CAP, "negative_cap": dense.NEGATIVE_CAP,
                    "first_mower": -dense.FIRST_MOWER_PENALTY,
                    "mower_used": -dense.LATER_MOWER_PENALTY, "win": 1.0, "loss": -1.0}}
        if smooth_total:
            settings["shaping_limit_mode"] = "smooth_total"
            settings["shaping_transform"] = "softsign_unit_slope_v1"
        return settings
    if scheme.name == "staged":
        from .staged_reward import (STAGE_TICKS, STAGE_CAP, RESOURCE_REFERENCE,
                                    STAGE_DAMAGE_WEIGHT, TERMINAL_DAMAGE_WEIGHT,
                                    MOWER_PENALTY, STAGE_LIFETIME_CAP, METRICS_VERSION)
        return {"version": 2, "scheme": "staged", "mower_basis": "native_trigger_counter_and_squish_v1", "coefficients": {
            "stage_ticks": STAGE_TICKS, "stage_cap": STAGE_CAP,
            "stage_lifetime_cap": STAGE_LIFETIME_CAP,
            "resource_reference": RESOURCE_REFERENCE,
            "stage_damage": STAGE_DAMAGE_WEIGHT, "terminal_damage": TERMINAL_DAMAGE_WEIGHT,
            "mower_used": -MOWER_PENALTY, "win": 1.0, "loss": -1.0,
        }, "metrics_version": METRICS_VERSION,
                "health_basis": "initial_native_adventure_spawn_plan"}
    coefficients = {name: float(getattr(scheme, name)) for name in (
        "terminal_scale", "mower_used", "plant_death", "zombie_kill",
        "wave_progress", "sun_collect", "zombie_threat", "terminal_plants", "terminal_sun",
    )}
    if scheme.name == "sparse":
        coefficients.update(terminal_scale=float(config.reward_scale),
                            mower_used=-abs(float(config.mower_penalty)))
    return {"version": 1, "scheme": scheme.name, "mower_basis": "native_trigger_counter_and_squish_v1", "coefficients": coefficients}


def reward_scheme_summary(
    name: str,
    *,
    sparse_mower_penalty: float | None = None,
) -> str:
    """Return a compact UI description of one reward preset."""
    scheme = get_reward_scheme(name)
    if scheme.name == "transition_dense":
        return "Transition dense: WIN +1 / LOSS -1 · first mower -0.50 / later -0.10 · net episode shaping soft bounds +/-0.40 · assets /5000 · damage 0.30 · threat 0.10 (step +/-0.01)"
    if scheme.name == "staged":
        return "Staged: WIN +1 / LOSS -1 · stage <=+0.10/1000 ticks (total <=+1.00) · mower -0.20 · end damage +0.50"
    mower = scheme.mower_used
    if scheme.name == "sparse" and sparse_mower_penalty is not None:
        mower = -abs(float(sparse_mower_penalty))
    return (
        f"{scheme.label}: WIN +1 · LOSS -1 · mower {mower:+.2f} · "
        f"threat ±{abs(scheme.zombie_threat):.2f} · "
        f"terminal plants +{scheme.terminal_plants:.2f} · sun +{scheme.terminal_sun:.2f}"
    )


def _event_count(events: Mapping[str, Any] | None, key: str) -> int:
    if not events:
        return 0
    try:
        return max(0, int(events.get(key, 0) or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _event_value(
    events: Mapping[str, Any] | None,
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if not events:
        return 0.0
    try:
        value = float(events.get(key, 0.0) or 0.0)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def training_reward(
    raw_reward: float,
    events: Mapping[str, Any] | None,
    config: "DoubleDQNConfig | None" = None,
    *,
    reward_scale: float | None = None,
    mower_penalty: float | None = None,
    reward_scheme: str | None = None,
) -> float:
    """Return the reward stored in replay for one transition.

    ``sparse`` preserves the existing configurable ``reward_scale`` and
    ``mower_penalty`` fields for checkpoint/backward compatibility. Named dense
    presets use their registered coefficients exactly.

    Balanced shaping consumes three bounded diagnostics exposed by ``PvZEnv``:
    ``zombie_threat_delta`` in [-1, 1], and terminal plant/sun scores in [0, 1].
    This keeps ordinary step rewards small while preserving terminal win/loss as
    the strongest single signal.
    """
    event_values = events
    if events and isinstance(events.get("events"), Mapping):
        event_values = events["events"]

    if config is not None:
        scheme = get_reward_scheme(config.reward_scheme)
        if scheme.name == "sparse":
            terminal_scale = float(config.reward_scale)
            mower_weight = -abs(float(config.mower_penalty))
        else:
            terminal_scale = scheme.terminal_scale
            mower_weight = scheme.mower_used
    elif reward_scheme is not None:
        scheme = get_reward_scheme(reward_scheme)
        terminal_scale = scheme.terminal_scale
        mower_weight = scheme.mower_used
    else:
        if reward_scale is None or mower_penalty is None:
            raise TypeError(
                "config, reward_scheme, or both reward_scale/mower_penalty are required"
            )
        scheme = REWARD_SCHEMES["sparse"]
        terminal_scale = float(reward_scale)
        mower_weight = -abs(float(mower_penalty))

    if scheme.name in ("staged", "transition_dense"):
        raise ValueError(f"{scheme.name} rewards require a per-episode tracker and raw game facts")
    reward = float(raw_reward) * terminal_scale
    reward += mower_weight * _event_count(event_values, "mowers_used")
    reward += scheme.plant_death * _event_count(event_values, "plants_lost")
    reward += scheme.zombie_kill * _event_count(event_values, "zombies_killed")
    reward += scheme.wave_progress * _event_count(event_values, "waves_advanced")
    reward += scheme.sun_collect * _event_count(event_values, "sun_collected")
    reward += scheme.zombie_threat * _event_value(
        event_values,
        "zombie_threat_delta",
        minimum=-1.0,
        maximum=1.0,
    )
    reward += scheme.terminal_plants * _event_value(
        event_values,
        "terminal_plant_score",
        minimum=0.0,
        maximum=1.0,
    )
    reward += scheme.terminal_sun * _event_value(
        event_values,
        "terminal_sun_score",
        minimum=0.0,
        maximum=1.0,
    )
    return float(reward)


def mower_event_count(events: Mapping[str, Any] | None) -> int:
    if events and isinstance(events.get("events"), Mapping):
        events = events["events"]
    return _event_count(events, "mowers_used")




__all__ = [
    "RewardScheme",
    "REWARD_SCHEMES",
    "REWARD_SCHEME_NAMES",
    "get_reward_scheme",
    "effective_reward_settings",
    "reward_scheme_summary",
    "training_reward",
    "mower_event_count",
]
