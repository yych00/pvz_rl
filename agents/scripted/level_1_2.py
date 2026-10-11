# -*- coding: utf-8 -*-
"""Adventure 1-2 policies for first-run and replay paths.

FIRST_RUN is the early campaign/tutorial path: use only cards actually present
in the battle seed bank and assume the minimal Peashooter + Sunflower kit.
REPLAY is the normal seed-chooser path and may use richer unlocked cards.

Formation rule shared with the improved 1-1 baseline:
- column 0 is the permanent attack column;
- Sunflowers start from column 1;
- all actions are still gated by the native authoritative action mask.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .level_1_1_policy import choose_level_1_1_plant

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

LEVEL_1_2_FIRST_RUN_DECK_PRIORITY = [SUNFLOWER, PEASHOOTER]
LEVEL_1_2_REPLAY_DECK_PRIORITY = [
    SUNFLOWER,
    PEASHOOTER,
    POTATO_MINE,
    WALL_NUT,
    CHERRY_BOMB,
    REPEATER,
    SNOW_PEA,
    SQUASH,
    JALAPENO,
    TORCHWOOD,
]
# Compatibility name used by the older 1-2 harness.
LEVEL_1_2_DECK_PRIORITY = list(LEVEL_1_2_REPLAY_DECK_PRIORITY)

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
    plant_type: int,
    rows: Sequence[int],
    cols: Sequence[int],
) -> Optional[Tuple[int, int, int, int, int]]:
    for row in rows:
        for col in cols:
            choice = _legal(mask, cards, plant_type, row, col)
            if choice is not None:
                return choice
    return None


def _snapshot(obs: dict, rows: int):
    row_plants: List[List[dict]] = [[] for _ in range(rows)]
    nearest = [9999.0] * rows
    zombie_counts = [0] * rows
    zombie_hp = [0] * rows
    sunflower_total = 0

    for plant in obs.get("plants", []):
        if not isinstance(plant, dict):
            continue
        row = int(plant.get("row", -1))
        col = int(plant.get("col", -1))
        plant_type = int(plant.get("type", -1))
        if 0 <= row < rows and 0 <= col < COLS:
            row_plants[row].append(plant)
            if plant_type == SUNFLOWER:
                sunflower_total += 1

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

    center_order = [r for r in (2, 1, 3, 0, 4, 5) if r < rows]
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
    return row_plants, nearest, zombie_counts, zombie_hp, sunflower_total, center_order, threat_order


def _count_type(plants: Sequence[dict], plant_type: int) -> int:
    return sum(1 for plant in plants if int(plant.get("type", -1)) == plant_type)


def _has_type(plants: Sequence[dict], plant_type: int) -> bool:
    return any(int(plant.get("type", -1)) == plant_type for plant in plants)


def _threat_score(row: int, nearest: Sequence[float], counts: Sequence[int], hp: Sequence[int]) -> float:
    return max(0.0, 760.0 - min(nearest[row], 760.0)) + counts[row] * 180.0 + min(hp[row], 1800) * 0.10


def choose_level_1_2_first_run_plant(mask: List[bool], obs: dict) -> Optional[Choice]:
    """Minimal first-run 1-2 policy: Peashooter attack line + Sunflower economy."""
    if len(mask) != ACTION_SPACE_SIZE:
        return None
    rows = max(1, min(MAX_ROWS, int(obs.get("global", {}).get("grid_rows", 5) or 5)))
    cards = _cards_by_type(obs)
    if not cards:
        return None

    row_plants, nearest, counts, hp, sunflower_total, center_order, threat_order = _snapshot(obs, rows)
    shooters = [_count_type(row_plants[r], PEASHOOTER) for r in range(rows)]

    # 1) Any live lane without damage gets column 0 first.
    active_uncovered = [
        r for r in threat_order
        if counts[r] > 0 and shooters[r] == 0 and nearest[r] < 740.0
    ]
    choice = _first_legal(mask, cards, PEASHOOTER, active_uncovered, (0,))
    if choice:
        return (*choice, "1-2-first-cover-col0")

    # 2) Close zombies get immediate extra DPS; no fragile economy greed here.
    danger = [
        r for r in threat_order
        if counts[r] > 0 and nearest[r] < 360.0 and shooters[r] < 3
    ]
    choice = _first_legal(mask, cards, PEASHOOTER, danger, (2, 3, 4, 5, 1))
    if choice:
        return (*choice, "1-2-first-critical-dps")

    # 3) Sunflower starts at column 1. Column 0 stays reserved for attackers.
    if sunflower_total < rows:
        safe_rows = [
            r for r in center_order
            if not _has_type(row_plants[r], SUNFLOWER)
            and (nearest[r] >= 420.0 or shooters[r] > 0)
        ]
        choice = _first_legal(mask, cards, SUNFLOWER, safe_rows, (1,))
        if choice:
            return (*choice, "1-2-first-sun-col1")

    # 4) Complete the permanent column-0 attack line on every lane.
    missing_attack = [r for r in threat_order if shooters[r] == 0]
    choice = _first_legal(mask, cards, PEASHOOTER, missing_attack, (0,))
    if choice:
        return (*choice, "1-2-first-attack-line")

    # 5) Small second economy layer in column 2 only when lanes are safe.
    economy_target = min(rows + 2, rows * 2)
    if sunflower_total < economy_target:
        economy_rows = sorted(
            range(rows),
            key=lambda r: (_count_type(row_plants[r], SUNFLOWER), _threat_score(r, nearest, counts, hp)),
        )
        safe_rows = [r for r in economy_rows if nearest[r] >= 440.0 or shooters[r] >= 2]
        choice = _first_legal(mask, cards, SUNFLOWER, safe_rows, (2,))
        if choice:
            return (*choice, "1-2-first-sun-col2")

    # 6) Convert the rest of the sun into second/third firing lines.
    for target_count, cols, reason in (
        (2, (3, 4, 5, 2), "1-2-first-second-line"),
        (3, (4, 5, 6, 3, 2), "1-2-first-third-line"),
    ):
        rows_need = sorted(
            (r for r in range(rows) if shooters[r] < target_count),
            key=lambda r: -_threat_score(r, nearest, counts, hp),
        )
        choice = _first_legal(mask, cards, PEASHOOTER, rows_need, cols)
        if choice:
            return (*choice, reason)

    hot = sorted(range(rows), key=lambda r: -_threat_score(r, nearest, counts, hp))
    choice = _first_legal(mask, cards, PEASHOOTER, hot, (5, 4, 6, 3, 7, 2, 8))
    if choice:
        return (*choice, "1-2-first-overflow")
    return None


def choose_level_1_2_replay_plant(mask: List[bool], obs: dict) -> Optional[Choice]:
    """Replay policy: use the richer 1-1 formation/tactical policy when possible."""
    choice = choose_level_1_1_plant(mask, obs)
    if choice is not None:
        idx, slot, row, col, plant_type, reason = choice
        return idx, slot, row, col, plant_type, f"1-2-replay:{reason}"
    return choose_level_1_2_first_run_plant(mask, obs)


def choose_level_1_2_plant(mask: List[bool], obs: dict) -> Optional[Choice]:
    """Compatibility entry point. Prefer replay-capable policy; it degrades safely."""
    return choose_level_1_2_replay_plant(mask, obs)
