"""Reward contracts for deck/seed selection, separate from battle reward."""

from __future__ import annotations


def seed_selection_action_reward(success: bool, *, invalid_penalty: float = -0.02) -> float:
    """Legal intermediate choices score zero; invalid actions get a tiny penalty."""
    return 0.0 if success else float(invalid_penalty)


def deck_outcome_reward(*, won: bool, lost: bool) -> float:
    """Delayed deck-level outcome reward assigned after the battle finishes."""
    if won:
        return 1.0
    if lost:
        return -1.0
    return 0.0
