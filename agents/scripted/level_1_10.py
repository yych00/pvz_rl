"""Adventure 1-10 conveyor policy: build a firing line and answer pressure."""

from __future__ import annotations

from typing import Optional, Sequence

from .conveyor import cards_by_type, lane_health, legal_choice, row_count, zombies_by_row
from .level_1_2 import ACTION_SPACE_SIZE, Choice

PEASHOOTER = 0
CHERRY_BOMB = 2
WALL_NUT = 3
POTATO_MINE = 4
SNOW_PEA = 5
CHOMPER = 6
REPEATER = 7
SHOOTERS = (REPEATER, SNOW_PEA, PEASHOOTER)


def _zombie_col(x: float) -> int:
    return max(0, min(8, int((x - 40) // 80)))


def choose_level_1_10_plant(mask: Sequence[bool], obs: dict) -> Optional[Choice]:
    if len(mask) != ACTION_SPACE_SIZE:
        return None
    rows = row_count(obs)
    cards = cards_by_type(obs)
    lanes = zombies_by_row(obs, rows)
    plants = [[] for _ in range(rows)]
    for plant in obs.get("plants", []):
        if isinstance(plant, dict):
            row = int(plant.get("row", -1))
            if 0 <= row < rows:
                plants[row].append(plant)
    shooter_count = [
        sum(int(p.get("type", -1)) in SHOOTERS for p in lane)
        for lane in plants
    ]
    wall_count = [
        sum(int(p.get("type", -1)) == WALL_NUT for p in lane)
        for lane in plants
    ]
    center = sorted(range(rows), key=lambda row: (abs(row - (rows - 1) / 2), row))
    threatened = sorted(
        (row for row in range(rows) if lanes[row]),
        key=lambda row: (
            float(lanes[row][0].get("x", 9999) or 9999),
            -len(lanes[row]),
            -lane_health(lanes[row]),
        ),
    )

    # Cherry is an emergency area attack. Place it near the closest zombie,
    # with adjacent rows as legal fallbacks for its 3x3 blast.
    for row in threatened:
        nearest = float(lanes[row][0].get("x", 9999) or 9999)
        if nearest >= 310 or (len(lanes[row]) < 2 and lane_health(lanes[row]) < 650):
            continue
        zcol = _zombie_col(nearest)
        blast_rows = [r for r in (row, row - 1, row + 1) if 0 <= r < rows]
        blast_cols = [c for c in (zcol - 1, zcol, zcol + 1) if 0 <= c < 9]
        choice = legal_choice(
            mask, cards, CHERRY_BOMB, blast_rows, blast_cols, "1-10-critical-cherry"
        )
        if choice:
            return choice

    # Every active lane needs a permanent shooter before adding support.
    missing = [row for row in threatened if shooter_count[row] == 0]
    for plant_type in SHOOTERS:
        choice = legal_choice(
            mask, cards, plant_type, missing, (0, 1, 2), "1-10-cover-live-lane"
        )
        if choice:
            return choice

    # Build the line even while no zombie is visible; conveyor cards have no
    # sun cost and holding a full belt slows generation of new packets.
    missing = [row for row in center if shooter_count[row] == 0]
    for plant_type in SHOOTERS:
        choice = legal_choice(
            mask, cards, plant_type, missing, (0, 1, 2), "1-10-build-firing-line"
        )
        if choice:
            return choice

    for row in threatened:
        nearest = float(lanes[row][0].get("x", 9999) or 9999)
        if nearest < 560 and wall_count[row] == 0:
            choice = legal_choice(
                mask, cards, WALL_NUT, (row,), (5, 4, 6, 3), "1-10-block-threat"
            )
            if choice:
                return choice

    # Spend short-range and delayed-effect cards while there is a suitable
    # target, so they do not accumulate and block new conveyor packets.
    for row in threatened:
        nearest = float(lanes[row][0].get("x", 9999) or 9999)
        if nearest > 440:
            choice = legal_choice(
                mask, cards, POTATO_MINE, (row,), (5, 6, 4), "1-10-mine"
            )
            if choice:
                return choice
        if nearest < 530:
            choice = legal_choice(
                mask, cards, CHOMPER, (row,), (4, 5, 3), "1-10-chomper"
            )
            if choice:
                return choice

    # Reinforce lanes by current threat, then by least existing firepower.
    reinforce = sorted(
        range(rows),
        key=lambda row: (
            0 if lanes[row] else 1,
            shooter_count[row],
            float(lanes[row][0].get("x", 9999) or 9999) if lanes[row] else 9999,
            center.index(row),
        ),
    )
    for plant_type in SHOOTERS:
        choice = legal_choice(
            mask, cards, plant_type, reinforce, (1, 2, 3, 4, 0), "1-10-reinforce"
        )
        if choice:
            return choice

    choice = legal_choice(
        mask, cards, WALL_NUT, center, (5, 4, 6, 3), "1-10-build-wall"
    )
    if choice:
        return choice
    return legal_choice(
        mask, cards, POTATO_MINE, center, (6, 5, 7), "1-10-preplant-mine"
    )


choose_level_1_10_first_run_plant = choose_level_1_10_plant
choose_level_1_10_replay_plant = choose_level_1_10_plant
