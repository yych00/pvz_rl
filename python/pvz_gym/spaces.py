"""Compatibility exports for the fixed Survival Endless spaces."""

from .action_space import (
    ACTION_SPACE_SIZE,
    COB_BASE,
    INTERACT_BASE,
    COLS,
    MAX_CARD_SLOTS,
    MAX_ROWS,
    NOOP,
    SHOVEL_NORMAL_BASE,
    SHOVEL_PUMPKIN_BASE,
    action_count,
    decode_action,
    encode_action,
    make_action_space,
)
from .observation import make_observation_space


def make_spaces(**_kwargs):
    return make_action_space(), make_observation_space()


__all__ = [
    "ACTION_SPACE_SIZE", "COB_BASE", "INTERACT_BASE", "COLS",
    "MAX_CARD_SLOTS", "MAX_ROWS", "NOOP", "SHOVEL_NORMAL_BASE",
    "SHOVEL_PUMPKIN_BASE",
    "action_count", "decode_action", "encode_action", "make_spaces",
]
