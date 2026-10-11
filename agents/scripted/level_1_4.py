# -*- coding: utf-8 -*-
"""Strong scripted policy for Adventure 1-4 LIVE/Gym validation.

Adventure-1 1-4 is intentionally limited to the plants genuinely available
before completing the level: Peashooter, Sunflower, Cherry Bomb and Wall-nut.
Potato Mine is the 1-4 reward, so the first-run policy never assumes it exists.

The policy is winning-oriented rather than minimal. It uses the native action
mask as the sole legality source and reacts to real observation facts: lane
distance, zombie count, body+helmet/shield HP, current wave and huge-wave state.
Replay mode may use the richer already-unlocked deck.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .level_1_1_policy import choose_level_1_1_plant
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
)

CHERRY_BOMB = 2
WALL_NUT = 3
POTATO_MINE = 4
SNOW_PEA = 5
REPEATER = 7
SQUASH = 17
JALAPENO = 20
TORCHWOOD = 22

# Entering first-run 1-4, these are the four plants actually unlocked.
LEVEL_1_4_FIRST_RUN_DECK_PRIORITY = [
    SUNFLOWER,
    PEASHOOTER,
    CHERRY_BOMB,
    WALL_NUT,
]

# Replay is deliberately strong while still degrading safely when a card is
# unavailable; the native SeedChooser/action masks remain authoritative.
LEVEL_1_4_REPLAY_DECK_PRIORITY = [
    SUNFLOWER,
    REPEATER,
    SNOW_PEA,
    PEASHOOTER,
    WALL_NUT,
    POTATO_MINE,
    CHERRY_BOMB,
    SQUASH,
    JALAPENO,
    TORCHWOOD,
]
LEVEL_1_4_DECK_PRIORITY = list(LEVEL_1_4_REPLAY_DECK_PRIORITY)

LAWN_X_MIN = 40
CELL_WIDTH = 80


def _zombie_col(x: float) -> int:
    return max(0, min(COLS - 1, int((float(x) - LAWN_X_MIN) // CELL_WIDTH)))


def _first_legal_type(
    mask: Sequence[bool],
    cards: dict[int, int],
    plant_type: int,
    rows: Sequence[int],
    cols: Sequence[int],
):
    for row in rows:
        for col in cols:
            choice = _legal(mask, cards, plant_type, int(row), int(col))
            if choice is not None:
                return choice
    return None


def _shield_snapshot(obs: dict, rows: int) -> tuple[List[int], List[int]]:
    """Return per-lane helmet/shield HP and armored-zombie counts."""
    shield_hp = [0] * rows
    armored_count = [0] * rows
    for zombie in obs.get("zombies", []):
        if not isinstance(zombie, dict):
            continue
        row = int(zombie.get("row", -1))
        if not 0 <= row < rows:
            continue
        shield = max(0, int(zombie.get("shield_hp", 0) or 0))
        shield += max(0, int(zombie.get("helm_hp", 0) or 0))
        shield_hp[row] += shield
        if shield > 0:
            armored_count[row] += 1
    return shield_hp, armored_count


def _count_type(plants: Sequence[dict], plant_type: int) -> int:
    return sum(1 for p in plants if int(p.get("type", -1)) == plant_type)


def _has_type(plants: Sequence[dict], plant_type: int) -> bool:
    return any(int(p.get("type", -1)) == plant_type for p in plants)


def _has_type_at_col(plants: Sequence[dict], plant_type: int, col: int) -> bool:
    return any(
        int(p.get("type", -1)) == plant_type and int(p.get("col", -1)) == col
        for p in plants
    )


def _wall_cols(x: float) -> List[int]:
    """Prefer a wall a few cells in front of the back-line formation."""
    zcol = _zombie_col(x)
    raw = [min(6, zcol - 2), min(5, zcol - 3), 6, 5, 4, 3, 2]
    result: List[int] = []
    for col in raw:
        col = max(2, min(COLS - 1, int(col)))
        # A wall at/behind a zombie is not an interception.
        if col >= zcol and zcol > 2:
            continue
        if col not in result:
            result.append(col)
    return result


def _cherry_cols(x: float) -> List[int]:
    zcol = _zombie_col(x)
    result: List[int] = []
    for col in (zcol - 1, zcol, zcol - 2, zcol + 1, 5, 4, 3):
        col = max(0, min(COLS - 1, int(col)))
        if col not in result:
            result.append(col)
    return result


def _cherry_choice(
    mask: Sequence[bool],
    cards: dict[int, int],
    focus_row: int,
    focus_x: float,
    rows: int,
):
    # Same-row first; adjacent rows are fallback target cells. Cherry's 3x3
    # blast makes any of these useful around a real cluster.
    target_rows = [
        r
        for r in (focus_row, focus_row - 1, focus_row + 1)
        if 0 <= r < rows
    ]
    return _first_legal_type(
        mask,
        cards,
        CHERRY_BOMB,
        target_rows,
        _cherry_cols(focus_x),
    )


def choose_level_1_4_first_run_plant(
    mask: List[bool],
    obs: dict,
) -> Optional[Choice]:
    """High-win-rate first-run 1-4 policy using only the genuine four-card kit."""
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
        nearest,
        counts,
        hp,
        sunflower_total,
        center_order,
        threat_order,
    ) = _snapshot(obs, rows)
    shield_hp, armored_count = _shield_snapshot(obs, rows)

    global_state = obs.get("global", {})
    current_wave = max(0, int(global_state.get("current_wave", 0) or 0))
    huge_wave = bool(global_state.get("is_huge_wave", False))
    total_zombies = sum(counts)
    total_shield = sum(shield_hp)

    peas = [_count_type(row_plants[r], PEASHOOTER) for r in range(rows)]
    walls = [_count_type(row_plants[r], WALL_NUT) for r in range(rows)]

    # 1) Emergency breach control. Cherry is reserved for genuinely expensive
    # threats; otherwise preserve the cooldown and buy time with Wall-nut/DPS.
    critical_rows = [
        r for r in threat_order
        if counts[r] > 0 and nearest[r] < 285.0
    ]
    for row in critical_rows:
        expensive = (
            armored_count[row] > 0
            or counts[row] >= 2
            or hp[row] >= 500
        )
        if expensive and CHERRY_BOMB in cards:
            choice = _cherry_choice(mask, cards, row, nearest[row], rows)
            if choice:
                return (*choice, "1-4-critical-cherry")

        if walls[row] == 0 and WALL_NUT in cards:
            choice = _first_legal_type(
                mask, cards, WALL_NUT, (row,), _wall_cols(nearest[row])
            )
            if choice:
                return (*choice, "1-4-critical-wall")

        choice = _first_legal_type(
            mask, cards, PEASHOOTER, (row,), (2, 3, 4, 5, 6, 1)
        )
        if choice:
            return (*choice, "1-4-critical-dps")

    # 2) Never leave a live lane completely uncovered. This outranks economy.
    uncovered = [
        r for r in threat_order
        if counts[r] > 0 and peas[r] == 0 and nearest[r] < 760.0
    ]
    choice = _first_legal_type(mask, cards, PEASHOOTER, uncovered, (0,))
    if choice:
        return (*choice, "1-4-cover-live-lane")

    # 3) Cone/armored pressure gets a second gun early. shield_hp is the native
    # observation fact, so this does not depend on hard-coded zombie type IDs.
    armored_pressure = [
        r for r in threat_order
        if counts[r] > 0
        and peas[r] < 2
        and nearest[r] < 700.0
        and (armored_count[r] > 0 or shield_hp[r] > 0 or hp[r] >= 500)
    ]
    choice = _first_legal_type(
        mask, cards, PEASHOOTER, armored_pressure, (2, 3, 4, 5, 6)
    )
    if choice:
        return (*choice, "1-4-armored-second-gun")

    # 4) Establish one Sunflower per lane. Rows under real pressure are delayed;
    # when the card is cooling down, return None instead of wasting sun just
    # because some other plant happens to be legal.
    if sunflower_total < rows:
        safe_rows = [
            r for r in center_order
            if not _has_type(row_plants[r], SUNFLOWER)
            and (
                counts[r] == 0
                or nearest[r] >= 470.0
                or peas[r] >= 2
            )
        ]
        choice = _first_legal_type(mask, cards, SUNFLOWER, safe_rows, (1,))
        if choice:
            return (*choice, "1-4-first-economy")

        if total_zombies == 0 or sunflower_total < 3:
            return None

    # 5) Put a real front line in lanes that are being asked to tank armor or
    # multiple bodies. Do not spam Wall-nuts into empty lanes.
    wall_rows = [
        r for r in threat_order
        if counts[r] > 0
        and walls[r] == 0
        and nearest[r] < 625.0
        and (
            armored_count[r] > 0
            or counts[r] >= 2
            or hp[r] >= 500
        )
    ]
    for row in wall_rows:
        choice = _first_legal_type(
            mask, cards, WALL_NUT, (row,), _wall_cols(nearest[row])
        )
        if choice:
            return (*choice, "1-4-pressure-wall")

    # 6) Cherry is the wave insurance. Use it for a true cluster/huge-wave
    # pressure event, not merely because the card cooled down.
    cluster_pressure = (
        total_zombies >= 6
        or any(counts[r] >= 3 for r in range(rows))
        or total_shield >= 900
        or (huge_wave and total_zombies >= 3)
    )
    if cluster_pressure and CHERRY_BOMB in cards:
        candidates = sorted(
            (r for r in range(rows) if counts[r] > 0),
            key=lambda r: -(
                _threat_score(r, nearest, counts, hp)
                + shield_hp[r] * 0.30
            ),
        )
        for row in candidates:
            choice = _cherry_choice(mask, cards, row, nearest[row], rows)
            if choice:
                return (*choice, "1-4-wave-cherry")

    # 7) Once the first economy row is established, build a permanent col-0
    # Peashooter line. Earlier we deliberately waited for actual zombies.
    if sunflower_total >= rows or current_wave >= 2:
        missing_attack = [
            r for r in threat_order
            if not _has_type_at_col(row_plants[r], PEASHOOTER, 0)
        ]
        choice = _first_legal_type(mask, cards, PEASHOOTER, missing_attack, (0,))
        if choice:
            return (*choice, "1-4-complete-first-gun-line")

    # 8) Modest second economy layer. 1-4 is only 10 waves, so stop at 7
    # Sunflowers and stop greed once the later waves are underway.
    economy_target = min(rows + 2, 7)
    if sunflower_total < economy_target and current_wave <= 6:
        economy_rows = sorted(
            range(rows),
            key=lambda r: (
                _count_type(row_plants[r], SUNFLOWER),
                _threat_score(r, nearest, counts, hp),
            ),
        )
        safe_rows = [
            r for r in economy_rows
            if counts[r] == 0
            or nearest[r] >= 520.0
            or peas[r] >= 2
        ]
        choice = _first_legal_type(
            mask, cards, SUNFLOWER, safe_rows, (2, 1)
        )
        if choice:
            return (*choice, "1-4-second-economy")

    # 9) Every lane gets two guns, highest-pressure lane first.
    need_second = [r for r in threat_order if peas[r] < 2]
    choice = _first_legal_type(
        mask, cards, PEASHOOTER, need_second, (3, 4, 2, 5, 6)
    )
    if choice:
        return (*choice, "1-4-second-gun-line")

    # 10) Once damage is established, even a single close zombie gets a wall.
    active_without_wall = [
        r for r in threat_order
        if counts[r] > 0 and walls[r] == 0 and nearest[r] < 670.0
    ]
    for row in active_without_wall:
        choice = _first_legal_type(
            mask, cards, WALL_NUT, (row,), _wall_cols(nearest[row])
        )
        if choice:
            return (*choice, "1-4-active-wall")

    # 11) Late-game third guns concentrate on the most dangerous lane first.
    reinforce = sorted(
        (r for r in range(rows) if peas[r] < 3),
        key=lambda r: -(
            _threat_score(r, nearest, counts, hp)
            + shield_hp[r] * 0.20
        ),
    )
    choice = _first_legal_type(
        mask, cards, PEASHOOTER, reinforce, (4, 5, 3, 6, 2, 7)
    )
    if choice:
        return (*choice, "1-4-third-gun-line")

    # 12) Late cluster clear, then convert spare sun into DPS.
    if CHERRY_BOMB in cards and total_zombies >= 4:
        candidates = [r for r in threat_order if counts[r] > 0]
        for row in candidates:
            choice = _cherry_choice(mask, cards, row, nearest[row], rows)
            if choice:
                return (*choice, "1-4-late-cherry")

    hot_rows = sorted(
        range(rows),
        key=lambda r: -(
            _threat_score(r, nearest, counts, hp)
            + shield_hp[r] * 0.15
        ),
    )
    choice = _first_legal_type(
        mask, cards, PEASHOOTER, hot_rows, (5, 6, 4, 7, 3, 8, 2)
    )
    if choice:
        return (*choice, "1-4-overflow-dps")

    return None


def choose_level_1_4_replay_plant(
    mask: List[bool],
    obs: dict,
) -> Optional[Choice]:
    """Replay 1-4: use the mature rich-deck policy, then degrade safely."""
    choice = choose_level_1_1_plant(mask, obs)
    if choice is not None:
        idx, slot, row, col, plant_type, reason = choice
        return idx, slot, row, col, plant_type, f"1-4-replay:{reason}"
    return choose_level_1_4_first_run_plant(mask, obs)


def choose_level_1_4_plant(mask: List[bool], obs: dict) -> Optional[Choice]:
    """Compatibility entry point; replay policy safely handles smaller decks."""
    return choose_level_1_4_replay_plant(mask, obs)
