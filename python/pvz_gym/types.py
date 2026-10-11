"""Python 侧使用的原生游戏事实数据结构，不绑定具体传输方式。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Mapping, Sequence


# Twenty minutes of game logic at the native 10 ms tick duration.
DEFAULT_MAX_EPISODE_TICKS = 120_000


class GameStatus(IntEnum):
    UNKNOWN = 0
    MENU = 1
    CHOOSING_SEEDS = 2
    PLAYING = 3
    WON = 4
    LOST = 5


class RngSeedSupport(IntEnum):
    UNSUPPORTED = 0
    FULL = 2


@dataclass(frozen=True)
class Action:
    type: str = "noop"
    card_slot: int = -1
    row: int = -1
    col: int = -1
    shovel_target: str = "normal"
    target_row: int = -1
    target_col: int = -1


@dataclass(frozen=True)
class ActionResult:
    success: bool
    error: int | str | None = None
    message: str = ""


@dataclass(frozen=True)
class StepEvents:
    zombies_killed: int = 0
    plants_lost: int = 0
    sun_collected: int = 0
    sun_spent: int = 0
    mowers_used: int = 0
    waves_advanced: int = 0
    won: bool = False
    lost: bool = False
    mower_triggered_rows: Sequence[int] = field(default_factory=tuple)


@dataclass(frozen=True)
class Observation:
    global_state: Mapping[str, object]
    cards: Sequence[Mapping[str, object]] = field(default_factory=tuple)
    plants: Sequence[Mapping[str, object]] = field(default_factory=tuple)
    zombies: Sequence[Mapping[str, object]] = field(default_factory=tuple)
    dropped_cards: Sequence[Mapping[str, object]] = field(default_factory=tuple)
    grid_items: Sequence[Mapping[str, object]] = field(default_factory=tuple)
    ice_trails: Sequence[Mapping[str, object]] = field(default_factory=tuple)
    mowers: Sequence[Mapping[str, object]] = field(default_factory=tuple)
    grid: Sequence[Mapping[str, object]] = field(default_factory=tuple)
    zombie_health: Sequence[Mapping[str, object]] = field(default_factory=tuple)
    boss_projectiles: Sequence[Mapping[str, object]] = field(default_factory=tuple)
    schema_version: int = 3
    quality: str = "fresh"


@dataclass(frozen=True)
class ResetFacts:
    observation: Observation | None
    game_clock: int
    status: GameStatus = GameStatus.UNKNOWN
    rng_seed_support: RngSeedSupport = RngSeedSupport.UNSUPPORTED
    rng_seed_applied: bool = False


@dataclass(frozen=True)
class StepFacts:
    observation: Observation
    events: StepEvents
    action: ActionResult
    action_mask: Sequence[bool]
    status: GameStatus
    game_clock: int
    automation_collected: int = 0
    terminal_observation_fallback: bool = False
    observation_quality: str = "fresh"
    last_confirmed_game_clock: int | None = None
    ticks_advanced: int | None = None
    ticks_advanced_exact: bool = True
    replay_eligible: bool = True
