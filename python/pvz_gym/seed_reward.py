"""Backward-compatible import path for seed/deck rewards."""
from .rewards.seed_selection import deck_outcome_reward, seed_selection_action_reward

__all__ = ["deck_outcome_reward", "seed_selection_action_reward"]
