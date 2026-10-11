"""Backward-compatible import path for battle rewards.

Canonical code lives in :mod:`pvz_gym.rewards.battle`.
"""
from .rewards.battle import RewardFn, calculate_reward, survival_reward

__all__ = ["RewardFn", "calculate_reward", "survival_reward"]
