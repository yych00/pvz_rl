# -*- coding: utf-8 -*-
"""Winning-oriented pure policy for Adventure 1-1 LIVE tests.

Formation rules used by this baseline:
- column 0 is reserved for permanent attacking plants;
- Sunflowers start from column 1, never column 0;
- Potato Mine remains part of the deck, but is only planted while the nearest
  zombie is still far enough away for the mine to arm safely;
- emergency threats are handled by immediate plants instead of late mines;
- every returned action is gated by the authoritative native action mask.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


ACTION_SPACE_SIZE = 757
PLANT_BASE = 1
MAX_CARD_SLOTS = 10
MAX_ROWS = 6
COLS = 9

PEASHOOTER = 0
SUNFLOWER = 1
CHERRY_BOMB = 2
WALL_NUT = 3
POTATO_MINE = 4
SNOW_PEA = 5
REPEATER = 7
SQUASH = 17
JALAPENO = 20
TORCHWOOD = 22

# Keep Potato Mine deliberately high in the deck priority.  The policy likes
# the card; the fix is better timing/placement, not removing it.
LEVEL_1_1_DECK_PRIORITY = [
    SUNFLOWER,
    PEASHOOTER,
    POTATO_MINE,
    WALL_NUT,
    REPEATER,
    SNOW_PEA,
    SQUASH,
    CHERRY_BOMB,
    JALAPENO,
    TORCHWOOD,
]
LEVEL_1_1_DECK_PRIORITY_V2 = list(LEVEL_1_1_DECK_PRIORITY)

SHOOTER_TYPES = (PEASHOOTER, REPEATER, SNOW_PEA)
PEA_TYPES = (PEASHOOTER, REPEATER)
INSTANT_TYPES = (SQUASH, JALAPENO, CHERRY_BOMB)

LAWN_X_MIN = 40
CELL_WIDTH = 80

Choice = Tuple[int, int, int, int, int, str]


def _cards_by_type(obs: dict) -> Dict[int, int]:
    cards: Dict[int, int] = {}
    for card in obs.get("cards", []):
        if not isinstance(card, dict):
            continue
        slot = int(card.get("slot", -1))
        plant_type = int(card.get("plant_type", -1))
        if 0 <= slot < MAX_CARD_SLOTS and plant_type >= 0:
            cards.setdefault(plant_type, slot)
    return cards


def _plant_action_index(slot: int, row: int, col: int) -> int:
    return PLANT_BASE + ((slot * MAX_ROWS + row) * COLS + col)


def _legal(
    mask: Sequence[bool],
    cards: Dict[int, int],
    plant_type: int,
    row: int,
    col: int,
) -> Optional[Tuple[int, int, int, int, int]]:
    slot = cards.get(int(plant_type))
    if slot is None or not (0 <= row < MAX_ROWS and 0 <= col < COLS):
        return None
    idx = _plant_action_index(slot, row, col)
    if idx >= len(mask) or not bool(mask[idx]):
        return None
    return idx, slot, row, col, int(plant_type)


def _first_legal(
    mask: Sequence[bool],
    cards: Dict[int, int],
    plant_types: Sequence[int],
    rows: Sequence[int],
    cols: Sequence[int],
) -> Optional[Tuple[int, int, int, int, int]]:
    for plant_type in plant_types:
        for row in rows:
            for col in cols:
                choice = _legal(mask, cards, plant_type, row, col)
                if choice is not None:
                    return choice
    return None


def _zombie_col(x: float) -> int:
    return max(0, min(COLS - 1, int((float(x) - LAWN_X_MIN) // CELL_WIDTH)))


def _front_cols(x: float, distance: int = 1) -> List[int]:
    zcol = _zombie_col(x)
    raw = [zcol - distance, zcol - distance - 1, zcol - 1, zcol, zcol + 1]
    cols: List[int] = []
    for col in raw:
        col = max(0, min(COLS - 1, int(col)))
        if col not in cols:
            cols.append(col)
    return cols


def _snapshot(obs: dict, rows: int):
    row_plants: List[List[dict]] = [[] for _ in range(rows)]
    totals: Dict[int, int] = {}
    nearest = [9999.0] * rows
    zombie_counts = [0] * rows
    zombie_hp = [0] * rows
    total_zombies = 0

    for plant in obs.get("plants", []):
        if not isinstance(plant, dict):
            continue
        row = int(plant.get("row", -1))
        col = int(plant.get("col", -1))
        plant_type = int(plant.get("type", -1))
        if 0 <= row < rows and 0 <= col < COLS:
            row_plants[row].append(plant)
            totals[plant_type] = totals.get(plant_type, 0) + 1

    for zombie in obs.get("zombies", []):
        if not isinstance(zombie, dict):
            continue
        row = int(zombie.get("row", -1))
        if not 0 <= row < rows:
            continue
        x = float(zombie.get("x", 9999.0) or 9999.0)
        hp = max(0, int(zombie.get("body_hp", zombie.get("hp", 0)) or 0)) + max(
            0, int(zombie.get("shield_hp", 0) or 0)
        ) + max(0, int(zombie.get("helm_hp", 0) or 0))
        nearest[row] = min(nearest[row], x)
        zombie_counts[row] += 1
        zombie_hp[row] += hp
        total_zombies += 1

    center_order = [row for row in (2, 1, 3, 0, 4, 5) if row < rows]
    center_rank = {row: i for i, row in enumerate(center_order)}
    threat_order = sorted(
        range(rows),
        key=lambda row: (
            nearest[row],
            -zombie_counts[row],
            -zombie_hp[row],
            center_rank.get(row, row),
        ),
    )
    return (
        row_plants,
        totals,
        nearest,
        zombie_counts,
        zombie_hp,
        total_zombies,
        center_order,
        threat_order,
    )


def _count_types(plants: Sequence[dict], types: Sequence[int]) -> int:
    wanted = set(types)
    return sum(1 for p in plants if int(p.get("type", -1)) in wanted)


def _has_type(plants: Sequence[dict], plant_type: int) -> bool:
    return any(int(p.get("type", -1)) == plant_type for p in plants)


def _has_at_col(plants: Sequence[dict], col: int) -> bool:
    return any(int(p.get("col", -1)) == col for p in plants)


def _threat_score(
    row: int,
    nearest: Sequence[float],
    zombie_counts: Sequence[int],
    zombie_hp: Sequence[int],
) -> float:
    return (
        max(0.0, 760.0 - min(nearest[row], 760.0))
        + zombie_counts[row] * 170.0
        + min(zombie_hp[row], 2400) * 0.12
    )


def _safe_potato_cols(x: float) -> List[int]:
    """Choose an arming position well ahead of a still-distant zombie.

    Mines are intentionally kept around the middle/front half of the lawn.  We
    do not plant them in the cell immediately in front of a zombie; that was the
    old behaviour that often let the zombie eat an unarmed mine.
    """
    zcol = _zombie_col(x)
    candidates = [min(5, zcol - 3), min(4, zcol - 4), 4, 5, 3]
    cols: List[int] = []
    for col in candidates:
        col = max(2, min(6, int(col)))
        if col < zcol and col not in cols:
            cols.append(col)
    return cols


def choose_level_1_1_plant(mask: List[bool], obs: dict) -> Optional[Choice]:
    """Choose one legal action using the final 1-1 formation policy."""
    if len(mask) != ACTION_SPACE_SIZE:
        return None

    rows = max(
        1,
        min(MAX_ROWS, int(obs.get("global", {}).get("grid_rows", 5) or 5)),
    )
    cards = _cards_by_type(obs)
    if not cards:
        return None

    (
        row_plants,
        totals,
        nearest,
        zombie_counts,
        zombie_hp,
        total_zombies,
        center_order,
        threat_order,
    ) = _snapshot(obs, rows)

    huge_wave = bool(obs.get("global", {}).get("is_huge_wave", False))
    shooter_count = [_count_types(row_plants[r], SHOOTER_TYPES) for r in range(rows)]
    sunflower_total = totals.get(SUNFLOWER, 0)

    # 1) Immediate breach: only instant plants are valid emergency tools.  A
    # Potato Mine is deliberately excluded here because it cannot arm in time.
    danger_rows = [
        row for row in threat_order
        if zombie_counts[row] > 0 and nearest[row] < 340.0
    ]
    for row in danger_rows:
        choice = _first_legal(
            mask, cards, INSTANT_TYPES, (row,), _front_cols(nearest[row], 1)
        )
        if choice:
            return (*choice, "1-1-critical-intercept")

    # 2) Huge/dense waves: deterministic lane/cluster clear before formation.
    overloaded = sorted(
        (row for row in range(rows) if zombie_counts[row] > 0),
        key=lambda r: -_threat_score(r, nearest, zombie_counts, zombie_hp),
    )
    if huge_wave or total_zombies >= 7 or any(zombie_counts[r] >= 3 for r in range(rows)):
        for row in overloaded:
            choice = _first_legal(
                mask, cards, (JALAPENO,), (row,), _front_cols(nearest[row], 0)
            )
            if choice:
                return (*choice, "1-1-row-clear")
        if overloaded:
            focus = overloaded[0]
            adjacent = tuple(r for r in (focus - 1, focus, focus + 1) if 0 <= r < rows)
            choice = _first_legal(
                mask, cards, (CHERRY_BOMB,), adjacent, _front_cols(nearest[focus], 0)
            )
            if choice:
                return (*choice, "1-1-cluster-clear")

    # 3) Column 0 is the permanent first attack column.  Cover a live lane
    # before spending more sun on economy.
    live_uncovered = [
        row for row in threat_order
        if zombie_counts[row] > 0 and shooter_count[row] == 0 and nearest[row] < 740.0
    ]
    choice = _first_legal(
        mask, cards, (PEASHOOTER, SNOW_PEA, REPEATER), live_uncovered, (0,)
    )
    if choice:
        return (*choice, "1-1-cover-live-col0")

    # 4) Potato Mine stays in the strategy.  Plant it EARLY, only while the
    # zombie is still far away, and leave several cells of travel time for
    # arming.  This replaces the old late 'cheap delay' behaviour.
    potato_rows = [
        row for row in threat_order
        if zombie_counts[row] > 0
        and nearest[row] >= 680.0
        and not _has_type(row_plants[row], POTATO_MINE)
    ]
    for row in potato_rows:
        cols = _safe_potato_cols(nearest[row])
        choice = _first_legal(mask, cards, (POTATO_MINE,), (row,), cols)
        if choice:
            return (*choice, "1-1-early-armed-potato")

    # 5) First economy column is column 1.  Column 0 is never used for Sunflower.
    if sunflower_total < rows:
        safe_rows = [
            row for row in center_order
            if not _has_type(row_plants[row], SUNFLOWER)
            and (nearest[row] >= 420.0 or shooter_count[row] > 0)
        ]
        choice = _first_legal(mask, cards, (SUNFLOWER,), safe_rows, (1,))
        if choice:
            return (*choice, "1-1-sunflower-col1")

    # 6) Complete the permanent col-0 attack line on every lane.
    need_col0 = [
        row for row in threat_order
        if not any(
            int(p.get("type", -1)) in SHOOTER_TYPES and int(p.get("col", -1)) == 0
            for p in row_plants[row]
        )
    ]
    choice = _first_legal(
        mask, cards, (PEASHOOTER, SNOW_PEA, REPEATER), need_col0, (0,)
    )
    if choice:
        return (*choice, "1-1-complete-col0-attack")

    # 7) Active lanes get a Wall-nut before we overbuild the backfield.
    for row in threat_order:
        if zombie_counts[row] <= 0 or _has_type(row_plants[row], WALL_NUT):
            continue
        if nearest[row] >= 650.0:
            continue
        wall_cols = [6, 5, 4, 3]
        choice = _first_legal(mask, cards, (WALL_NUT,), (row,), wall_cols)
        if choice:
            return (*choice, "1-1-wall")

    # 8) Small second economy layer starts at column 2.  We still keep column 0
    # exclusively offensive.
    economy_target = min(rows + 2, rows * 2)
    if sunflower_total < economy_target:
        economy_rows = sorted(
            range(rows),
            key=lambda r: (
                _count_types(row_plants[r], (SUNFLOWER,)),
                _threat_score(r, nearest, zombie_counts, zombie_hp),
            ),
        )
        safe_rows = [r for r in economy_rows if nearest[r] >= 450.0 or shooter_count[r] >= 2]
        choice = _first_legal(mask, cards, (SUNFLOWER,), safe_rows, (2, 1))
        if choice:
            return (*choice, "1-1-second-economy")

    # 9) Second and third damage lines.  Threatened lanes get the expensive
    # cards first; column 0 remains intact as the first permanent attack line.
    need_second = [r for r in threat_order if shooter_count[r] < 2]
    choice = _first_legal(
        mask, cards, (REPEATER, SNOW_PEA, PEASHOOTER), need_second, (3, 4, 2, 5)
    )
    if choice:
        return (*choice, "1-1-second-line")

    # 10) Torchwood only in normal pea lanes; do not destroy Snow Pea utility.
    torch_rows: List[int] = []
    for row in threat_order:
        plants = row_plants[row]
        if _has_type(plants, TORCHWOOD) or _has_type(plants, SNOW_PEA):
            continue
        if any(
            int(p.get("type", -1)) in PEA_TYPES and int(p.get("col", -1)) < 5
            for p in plants
        ):
            torch_rows.append(row)
    choice = _first_legal(mask, cards, (TORCHWOOD,), torch_rows, (5, 4))
    if choice:
        return (*choice, "1-1-torchwood")

    reinforce = sorted(
        (r for r in range(rows) if shooter_count[r] < 3),
        key=lambda r: -_threat_score(r, nearest, zombie_counts, zombie_hp),
    )
    choice = _first_legal(
        mask, cards, (REPEATER, SNOW_PEA, PEASHOOTER), reinforce, (4, 5, 3, 6)
    )
    if choice:
        return (*choice, "1-1-third-line")

    # 11) Proactive late mines are also allowed, but only in quiet/far lanes.
    for row in threat_order:
        if zombie_counts[row] <= 0 or nearest[row] < 680.0:
            continue
        choice = _first_legal(mask, cards, (POTATO_MINE,), (row,), _safe_potato_cols(nearest[row]))
        if choice:
            return (*choice, "1-1-late-safe-potato")

    # 12) Finish missing walls, then overflow DPS.
    no_wall = [r for r in threat_order if not _has_type(row_plants[r], WALL_NUT)]
    choice = _first_legal(mask, cards, (WALL_NUT,), no_wall, (6, 5, 7))
    if choice:
        return (*choice, "1-1-wall-complete")

    hot_rows = sorted(
        range(rows), key=lambda r: -_threat_score(r, nearest, zombie_counts, zombie_hp)
    )
    choice = _first_legal(
        mask, cards, (REPEATER, SNOW_PEA, PEASHOOTER), hot_rows, (4, 5, 3, 6, 7, 8)
    )
    if choice:
        return (*choice, "1-1-overflow-dps")

    return None


# Compatibility name used by older tests/imports.
def choose_level_1_1_plant_v2(mask: List[bool], obs: dict) -> Optional[Choice]:
    return choose_level_1_1_plant(mask, obs)
