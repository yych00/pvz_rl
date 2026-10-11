"""战斗阶段固定的 757 维动作空间。"""

from __future__ import annotations

from .types import Action

NOOP = 0
MAX_CARD_SLOTS = 10
MAX_ROWS = 6
COLS = 9
PLANT_BASE = 1
PLANT_ACTIONS = MAX_CARD_SLOTS * MAX_ROWS * COLS
SHOVEL_NORMAL_BASE = PLANT_BASE + PLANT_ACTIONS
SHOVEL_CELLS = MAX_ROWS * COLS
SHOVEL_PUMPKIN_BASE = SHOVEL_NORMAL_BASE + SHOVEL_CELLS
SHOVEL_ACTIONS = 2 * SHOVEL_CELLS
COB_BASE = SHOVEL_NORMAL_BASE + SHOVEL_ACTIONS
COB_TARGET_CELLS = MAX_ROWS * COLS
COB_ACTIONS = COB_TARGET_CELLS
INTERACT_BASE = COB_BASE + COB_ACTIONS
INTERACT_ACTIONS = MAX_ROWS * COLS
ACTION_SPACE_SIZE = INTERACT_BASE + INTERACT_ACTIONS


def encode_action(action: Action | None = None, *, card_slot: int | None = None,
                  row: int | None = None, col: int | None = None) -> int:
    """把语义动作编码成稳定的 Gym 离散动作编号。"""
    if action is None and card_slot is None and row is None and col is None:
        return NOOP
    if action is not None:
        if action.type == "noop":
            return NOOP
        if action.type == "plant":
            card_slot, row, col = action.card_slot, action.row, action.col
        elif action.type == "shovel":
            _validate_cell(action.row, action.col)
            if action.shovel_target == "normal":
                base = SHOVEL_NORMAL_BASE
            elif action.shovel_target == "pumpkin":
                base = SHOVEL_PUMPKIN_BASE
            else:
                raise ValueError(f"unsupported shovel target: {action.shovel_target}")
            return base + action.row * COLS + action.col
        elif action.type == "cob_fire":
            if action.row != -1 or action.col != -1:
                raise ValueError("cob_fire no longer accepts a source cannon")
            _validate_cell(action.target_row, action.target_col)
            target = action.target_row * COLS + action.target_col
            return COB_BASE + target
        elif action.type == "interact_cell":
            _validate_cell(action.row, action.col)
            return INTERACT_BASE + action.row * COLS + action.col
        else:
            raise ValueError(f"unsupported battle action type: {action.type}")
    if card_slot is None or row is None or col is None:
        raise ValueError("plant action requires card_slot, row, and col")
    if not (0 <= card_slot < MAX_CARD_SLOTS):
        raise ValueError(f"card_slot out of range: {card_slot}")
    _validate_cell(row, col)
    return PLANT_BASE + ((card_slot * MAX_ROWS + row) * COLS + col)


def decode_action(action_id: int) -> Action:
    """把 Gym 动作编号还原为完整语义动作，不丢失空间参数。"""
    if isinstance(action_id, bool) or not isinstance(action_id, int):
        raise ValueError(f"action ID must be an int, got {type(action_id).__name__}")
    if action_id == NOOP:
        return Action()
    if not (PLANT_BASE <= action_id < ACTION_SPACE_SIZE):
        raise ValueError(f"action ID out of range: {action_id}")
    if action_id < SHOVEL_NORMAL_BASE:
        offset = action_id - PLANT_BASE
        card_slot, cell = divmod(offset, MAX_ROWS * COLS)
        row, col = divmod(cell, COLS)
        return Action("plant", card_slot, row, col)
    if action_id < COB_BASE:
        if action_id < SHOVEL_PUMPKIN_BASE:
            offset = action_id - SHOVEL_NORMAL_BASE
            target = "normal"
        else:
            offset = action_id - SHOVEL_PUMPKIN_BASE
            target = "pumpkin"
        row, col = divmod(offset, COLS)
        return Action("shovel", row=row, col=col, shovel_target=target)

    if action_id < INTERACT_BASE:
        target_row, target_col = divmod(action_id - COB_BASE, COLS)
        return Action("cob_fire", target_row=target_row, target_col=target_col)
    row, col = divmod(action_id - INTERACT_BASE, COLS)
    return Action("interact_cell", row=row, col=col)


def _validate_cell(row: int, col: int) -> None:
    if not (0 <= row < MAX_ROWS and 0 <= col < COLS):
        raise ValueError(f"grid coordinate out of range: ({row}, {col})")


def action_count(*_args, **_kwargs) -> int:
    return ACTION_SPACE_SIZE


def make_action_space():
    try:
        from gymnasium import spaces
        return spaces.Discrete(ACTION_SPACE_SIZE)
    except ImportError:  # pragma: no cover
        class Discrete:
            def __init__(self, n: int): self.n = n
            def contains(self, value: object) -> bool:
                return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < self.n
            def __eq__(self, other: object) -> bool:
                return hasattr(other, "n") and self.n == other.n
        return Discrete(ACTION_SPACE_SIZE)
