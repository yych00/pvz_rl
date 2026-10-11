"""Replaceable battle reward functions."""

from __future__ import annotations

from typing import Protocol

from ..types import GameStatus


class RewardFn(Protocol):
    def __call__(self, status: GameStatus) -> float: ...


def calculate_reward(status: GameStatus) -> float:
    if status == GameStatus.WON:
        return 100.0
    if status == GameStatus.LOST:
        return -100.0
    return 0.0


# Compatibility name. The minimal milestone intentionally has no shaping.
survival_reward = calculate_reward
