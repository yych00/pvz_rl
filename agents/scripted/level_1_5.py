"""Adventure 1-5 Wall-nut Bowling policy for the conveyor seed bank."""

from __future__ import annotations

from typing import Optional, Sequence

from .conveyor import cards_by_type, lane_health, legal_choice, row_count, zombies_by_row
from .level_1_2 import ACTION_SPACE_SIZE, Choice

WALL_NUT = 3
EXPLODE_O_NUT = 49


def choose_level_1_5_plant(mask: Sequence[bool], obs: dict) -> Optional[Choice]:
    if len(mask) != ACTION_SPACE_SIZE:
        return None
    rows = row_count(obs)
    cards = cards_by_type(obs)
    lanes = zombies_by_row(obs, rows)
    # A bowling nut may only be launched from columns 0..2. Column 2 gives
    # the ball the shortest travel without crossing the original red line.
    launch_cols = (2, 1, 0)
    threatened = sorted(
        (row for row in range(rows) if lanes[row]),
        key=lambda row: (
            float(lanes[row][0].get("x", 9999) or 9999),
            -len(lanes[row]),
            -lane_health(lanes[row]),
        ),
    )
    if not threatened:
        return None

    for row in threatened:
        nearest = float(lanes[row][0].get("x", 9999) or 9999)
        if nearest < 280 or len(lanes[row]) >= 3:
            choice = legal_choice(
                mask, cards, EXPLODE_O_NUT, (row,), launch_cols, "1-5-explosive-pressure"
            )
            if choice:
                return choice

    # Ordinary nuts are plentiful; target the nearest live lane first.
    choice = legal_choice(
        mask, cards, WALL_NUT, threatened, launch_cols, "1-5-bowl-threatened-lane"
    )
    if choice:
        return choice

    # Use an available explosive if a normal nut is absent. Wait until the
    # target has entered the board far enough that the blast is useful.
    close_rows = [
        row for row in threatened
        if float(lanes[row][0].get("x", 9999) or 9999) < 520
    ]
    return legal_choice(
        mask, cards, EXPLODE_O_NUT, close_rows, launch_cols, "1-5-bowl-explosive"
    )


choose_level_1_5_first_run_plant = choose_level_1_5_plant
choose_level_1_5_replay_plant = choose_level_1_5_plant
