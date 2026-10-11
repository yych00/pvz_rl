# -*- coding: utf-8 -*-
"""Adventure-1 / Adventure-2 policies for level 1-1.

Adventure 1 is the real tutorial-era 1-1 and therefore assumes only the
Peashooter that is actually present in the seed bank. Adventure 2+ uses the
full replay-capable 1-1 policy.

Both variants remain fully gated by the native authoritative action mask.
"""

from __future__ import annotations

from typing import List, Optional

from .level_1_1_policy import (
    ACTION_SPACE_SIZE,
    MAX_ROWS,
    PEASHOOTER,
    LEVEL_1_1_DECK_PRIORITY,
    Choice,
    _cards_by_type,
    _first_legal,
    _snapshot,
    _threat_score,
    choose_level_1_1_plant,
)


LEVEL_1_1_ADVENTURE1_DECK_PRIORITY = [PEASHOOTER]
LEVEL_1_1_ADVENTURE2_DECK_PRIORITY = list(LEVEL_1_1_DECK_PRIORITY)


def choose_level_1_1_adventure1_plant(
    mask: List[bool], obs: dict
) -> Optional[Choice]:
    """Tutorial-safe Adventure-1 policy using only Peashooter.

    Column 0 is preferred as the permanent attack column, but the policy falls
    back to any legal cell because the original 1-1 tutorial may temporarily
    constrain placement through its own action legality.
    """
    if len(mask) != ACTION_SPACE_SIZE:
        return None

    rows = max(
        1,
        min(MAX_ROWS, int(obs.get("global", {}).get("grid_rows", 5) or 5)),
    )
    cards = _cards_by_type(obs)
    if PEASHOOTER not in cards:
        return None

    (
        row_plants,
        _totals,
        nearest,
        zombie_counts,
        zombie_hp,
        _total_zombies,
        center_order,
        threat_order,
    ) = _snapshot(obs, rows)

    shooter_count = [
        sum(1 for p in row_plants[row] if int(p.get("type", -1)) == PEASHOOTER)
        for row in range(rows)
    ]

    # 1) Never leave an active lane without damage. Prefer the unified col-0
    # formation, then accept the first tutorial-legal fallback cell.
    uncovered = [
        row
        for row in threat_order
        if zombie_counts[row] > 0 and shooter_count[row] == 0
    ]
    choice = _first_legal(
        mask,
        cards,
        (PEASHOOTER,),
        uncovered,
        (0, 1, 2, 3, 4, 5, 6, 7, 8),
    )
    if choice:
        return (*choice, "1-1-adventure1-cover")

    # 2) Close zombies get immediate extra DPS.
    danger = [
        row
        for row in threat_order
        if zombie_counts[row] > 0
        and nearest[row] < 380.0
        and shooter_count[row] < 3
    ]
    choice = _first_legal(
        mask,
        cards,
        (PEASHOOTER,),
        danger,
        (1, 2, 3, 4, 5, 6, 7, 8, 0),
    )
    if choice:
        return (*choice, "1-1-adventure1-critical-dps")

    # 3) Complete a first attack line wherever the native tutorial legality
    # permits. Quiet center lanes are preferred when no zombie is active.
    missing = [row for row in center_order if shooter_count[row] == 0]
    choice = _first_legal(
        mask,
        cards,
        (PEASHOOTER,),
        missing,
        (0, 1, 2, 3, 4, 5, 6, 7, 8),
    )
    if choice:
        return (*choice, "1-1-adventure1-first-line")

    # 4) Spend surplus sun on the most pressured lane.
    hot_rows = sorted(
        range(rows),
        key=lambda row: -_threat_score(
            row, nearest, zombie_counts, zombie_hp
        ),
    )
    choice = _first_legal(
        mask,
        cards,
        (PEASHOOTER,),
        hot_rows,
        (1, 2, 3, 4, 5, 6, 7, 8, 0),
    )
    if choice:
        return (*choice, "1-1-adventure1-overflow")

    return None


def choose_level_1_1_adventure2_plant(
    mask: List[bool], obs: dict
) -> Optional[Choice]:
    """Adventure-2+ 1-1 policy with the full unlocked replay deck."""
    choice = choose_level_1_1_plant(mask, obs)
    if choice is None:
        return None
    idx, slot, row, col, plant_type, reason = choice
    return idx, slot, row, col, plant_type, f"1-1-adventure2:{reason}"
