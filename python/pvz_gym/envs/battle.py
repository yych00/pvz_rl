"""Compatibility name for the minimal battle Gym environment."""

from __future__ import annotations

from ..env import PvzGymEnv
from ..rewards.battle import calculate_reward


class BattleEnv(PvzGymEnv):
    def __init__(self, *args, reward_fn=calculate_reward, **kwargs):
        super().__init__(*args, reward_fn=reward_fn, **kwargs)
