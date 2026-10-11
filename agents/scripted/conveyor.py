"""Small helpers shared by Adventure conveyor-belt policies."""

from __future__ import annotations

from typing import Optional, Sequence

from .level_1_2 import ACTION_SPACE_SIZE, COLS, MAX_CARD_SLOTS, MAX_ROWS, Choice


def row_count(obs: dict) -> int:
    return max(1, min(MAX_ROWS, int(obs.get("global", {}).get("grid_rows", 5) or 5)))


def is_conveyor(obs: dict) -> bool:
    """Return True if the current battle operates on a conveyor belt."""
    global_info = obs.get("global", {})
    if "is_conveyor" in global_info:
        return bool(global_info.get("is_conveyor"))
    return str(global_info.get("card_system", "")) == "conveyor"


def card_system(obs: dict) -> str:
    """Return current card system mode: 'conveyor', 'dropped', or 'seed_bank'."""
    return str(obs.get("global", {}).get("card_system", "") or ("conveyor" if is_conveyor(obs) else "seed_bank"))


def game_mode_name(obs: dict) -> str:
    """Return human-readable mode name (e.g. 'Adventure 1-5', 'Wall-nut Bowling')."""
    return str(obs.get("global", {}).get("game_mode_name", "") or "")



def cards_by_type(obs: dict) -> dict[int, list[int]]:
    """Keep every packet: a conveyor can contain several of the same plant."""
    cards: dict[int, list[int]] = {}
    for card in obs.get("cards", []):
        if not isinstance(card, dict):
            continue
        slot = int(card.get("slot", -1))
        plant_type = int(card.get("plant_type", -1))
        if 0 <= slot < MAX_CARD_SLOTS and plant_type >= 0:
            cards.setdefault(plant_type, []).append(slot)
    return cards


def legal_choice(
    mask: Sequence[bool],
    cards: dict[int, list[int]],
    plant_type: int,
    rows: Sequence[int],
    cols: Sequence[int],
    reason: str,
) -> Optional[Choice]:
    if len(mask) != ACTION_SPACE_SIZE:
        return None
    for row in rows:
        if not 0 <= row < MAX_ROWS:
            continue
        for col in cols:
            if not 0 <= col < COLS:
                continue
            for slot in cards.get(plant_type, []):
                idx = 1 + ((slot * MAX_ROWS + row) * COLS + col)
                if mask[idx]:
                    return idx, slot, row, col, plant_type, reason
    return None


def zombies_by_row(obs: dict, rows: int) -> list[list[dict]]:
    lanes: list[list[dict]] = [[] for _ in range(rows)]
    for zombie in obs.get("zombies", []):
        if isinstance(zombie, dict):
            row = int(zombie.get("row", -1))
            if 0 <= row < rows:
                lanes[row].append(zombie)
    for lane in lanes:
        lane.sort(key=lambda zombie: float(zombie.get("x", 9999) or 9999))
    return lanes


def lane_health(lane: Sequence[dict]) -> int:
    return sum(
        max(0, int(zombie.get("body_hp", zombie.get("hp", 0)) or 0))
        + max(0, int(zombie.get("helm_hp", 0) or 0))
        + max(0, int(zombie.get("shield_hp", 0) or 0))
        for zombie in lane
    )
