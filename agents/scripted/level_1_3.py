# -*- coding: utf-8 -*-
"""Adventure 1-3 policies for first-run and replay paths.

The first-run policy assumes the early campaign kit and uses Cherry Bomb only
when it is actually present in the battle seed bank.  The replay policy can use
richer unlocked cards while preserving the same formation rule:
- column 0 is reserved for permanent attacking plants;
- Sunflowers begin at column 1;
- emergency plants are tactical, not part of the permanent back line;
- native action_mask is always authoritative.
"""

from __future__ import annotations

from typing import List, Optional

from .level_1_1_policy import CHERRY_BOMB, choose_level_1_1_plant
from .level_1_2_policy import (
    ACTION_SPACE_SIZE,
    COLS,
    MAX_ROWS,
    PEASHOOTER,
    SUNFLOWER,
    Choice,
    _cards_by_type,
    _legal,
    _snapshot,
    _threat_score,
    choose_level_1_2_first_run_plant,
)

WALL_NUT = 3
POTATO_MINE = 4
SNOW_PEA = 5
REPEATER = 7
SQUASH = 17
JALAPENO = 20
TORCHWOOD = 22

LEVEL_1_3_FIRST_RUN_DECK_PRIORITY = [SUNFLOWER, PEASHOOTER, CHERRY_BOMB]
LEVEL_1_3_REPLAY_DECK_PRIORITY = [
    SUNFLOWER,
    PEASHOOTER,
    CHERRY_BOMB,
    WALL_NUT,
    POTATO_MINE,
    REPEATER,
    SNOW_PEA,
    SQUASH,
    JALAPENO,
    TORCHWOOD,
]
LEVEL_1_3_DECK_PRIORITY = list(LEVEL_1_3_REPLAY_DECK_PRIORITY)

LAWN_X_MIN = 40
CELL_WIDTH = 80


def _zombie_col(x: float) -> int:
    return max(0, min(COLS - 1, int((float(x) - LAWN_X_MIN) // CELL_WIDTH)))


def _cherry_cols(x: float) -> List[int]:
    zcol = _zombie_col(x)
    cols: List[int] = []
    for col in (zcol - 1, zcol, zcol - 2, zcol + 1, 5, 4, 3):
        col = max(0, min(COLS - 1, int(col)))
        if col not in cols:
            cols.append(col)
    return cols


def choose_level_1_3_first_run_plant(mask: List[bool], obs: dict) -> Optional[Choice]:
    """First-run 1-3: Cherry tactical layer over the stable 1-2 formation."""
    if len(mask) != ACTION_SPACE_SIZE:
        return None

    rows = max(1, min(MAX_ROWS, int(obs.get("global", {}).get("grid_rows", 5) or 5)))
    cards = _cards_by_type(obs)
    if not cards:
        return None

    row_plants, nearest, counts, hp, _, _, threat_order = _snapshot(obs, rows)
    total_zombies = sum(counts)
    huge_wave = bool(obs.get("global", {}).get("is_huge_wave", False))

    # Cherry is the defining new tactical option for this early level.  Use it
    # only for real pressure: a close breach, a 2+ zombie lane, or a huge wave.
    if CHERRY_BOMB in cards:
        candidate_rows = sorted(
            (r for r in range(rows) if counts[r] > 0),
            key=lambda r: -_threat_score(r, nearest, counts, hp),
        )
        for row in candidate_rows:
            if not (
                nearest[row] < 310.0
                or counts[row] >= 2
                or huge_wave
                or total_zombies >= 6
            ):
                continue
            adjacent = [r for r in (row - 1, row, row + 1) if 0 <= r < rows]
            for target_row in adjacent:
                for col in _cherry_cols(nearest[row]):
                    choice = _legal(mask, cards, CHERRY_BOMB, target_row, col)
                    if choice is not None:
                        return (*choice, "1-3-first-cherry-control")

    # Otherwise use the disciplined Peashooter/Sunflower first-run formation.
    choice = choose_level_1_2_first_run_plant(mask, obs)
    if choice is not None:
        idx, slot, row, col, plant_type, reason = choice
        return idx, slot, row, col, plant_type, reason.replace("1-2-first", "1-3-first")
    return None


def choose_level_1_3_replay_plant(mask: List[bool], obs: dict) -> Optional[Choice]:
    """Replay 1-3: richer tactical deck, degrading to first-run policy if needed."""
    choice = choose_level_1_1_plant(mask, obs)
    if choice is not None:
        idx, slot, row, col, plant_type, reason = choice
        return idx, slot, row, col, plant_type, f"1-3-replay:{reason}"
    return choose_level_1_3_first_run_plant(mask, obs)


def choose_level_1_3_plant(mask: List[bool], obs: dict) -> Optional[Choice]:
    """Compatibility entry point."""
    return choose_level_1_3_replay_plant(mask, obs)
