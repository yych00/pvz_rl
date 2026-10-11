"""Transport types and action encoding for the seed-selection phase."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Sequence


class SeedSelectionActionType(IntEnum):
    SELECT = 0
    DESELECT = 1
    CONFIRM = 2
    SELECT_IMITATER = 3


SEED_TYPE_CAPACITY = 64
IMITATER_TYPE = 48
IMITATER_TARGET_COUNT = 40
SELECT_BASE = 0
IMITATER_BASE = SEED_TYPE_CAPACITY
DESELECT_BASE = 2 * SEED_TYPE_CAPACITY
CONFIRM_ID = 3 * SEED_TYPE_CAPACITY
SEED_ACTION_COUNT = CONFIRM_ID + 1


@dataclass(frozen=True)
class SeedOptionState:
    plant_type: int
    available: bool = False
    unlocked: bool = False
    selected: bool = False
    disabled: bool = False
    chooser_state: int = 0


class SeedWarning(IntEnum):
    NONE = 0
    NO_SUN_PRODUCER = 1 << 0
    NO_POOL_SUPPORT = 1 << 1
    NO_FLOWER_POT = 1 << 2
    NO_FLYER_DEFENSE = 1 << 3
    UPGRADE_BASE_MISSING = 1 << 4


@dataclass(frozen=True)
class SeedSelectionState:
    present: bool = False
    required_count: int = 0
    selected_count: int = 0
    can_confirm: bool = False
    seeds_in_flight: bool = False
    warning_flags: int = 0
    imitater_target: int = -1
    selected_seeds: Sequence[int] = field(default_factory=tuple)
    options: Sequence[SeedOptionState] = field(default_factory=tuple)
    # None means the DLL cannot provide the preview, not that the level is empty.
    zombie_types_in_level: Sequence[int] | None = None


@dataclass(frozen=True)
class SeedSelectionResult:
    success: bool
    error_code: int = 0
    message: str = ""


def seed_action_count(seed_type_capacity: int = SEED_TYPE_CAPACITY) -> int:
    if seed_type_capacity != SEED_TYPE_CAPACITY:
        raise ValueError(f"seed action codec requires capacity {SEED_TYPE_CAPACITY}")
    return SEED_ACTION_COUNT


def encode_seed_action(action_type: SeedSelectionActionType, plant_type: int | None = None) -> int:
    """Encode the version-one, fixed-width seed action ABI."""
    if action_type == SeedSelectionActionType.CONFIRM:
        if plant_type is not None:
            raise ValueError("confirm action has no plant type")
        return CONFIRM_ID
    if isinstance(plant_type, bool) or not isinstance(plant_type, int):
        raise ValueError("seed action requires an integer plant type")
    if not 0 <= plant_type < SEED_TYPE_CAPACITY:
        raise ValueError(f"plant type out of range: {plant_type}")
    if action_type == SeedSelectionActionType.SELECT:
        return SELECT_BASE + plant_type
    if action_type == SeedSelectionActionType.SELECT_IMITATER:
        return IMITATER_BASE + plant_type
    if action_type == SeedSelectionActionType.DESELECT:
        return DESELECT_BASE + plant_type
    raise ValueError(f"unsupported seed action type: {action_type}")


def decode_seed_action(value: int, seed_type_capacity: int) -> tuple[SeedSelectionActionType, int | None]:
    seed_action_count(seed_type_capacity)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"seed-selection action must be an int: {value!r}")
    if SELECT_BASE <= value < IMITATER_BASE:
        return SeedSelectionActionType.SELECT, value
    if IMITATER_BASE <= value < DESELECT_BASE:
        return SeedSelectionActionType.SELECT_IMITATER, value - IMITATER_BASE
    if DESELECT_BASE <= value < CONFIRM_ID:
        return SeedSelectionActionType.DESELECT, value - DESELECT_BASE
    if value == CONFIRM_ID:
        return SeedSelectionActionType.CONFIRM, None
    raise ValueError(f"seed-selection action out of range: {value}")
