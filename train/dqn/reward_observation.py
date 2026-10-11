"""DQN-only reward context; the algorithm-neutral Gym schema stays unchanged.

Historical budgets and new net-balance headroom share the three-input shape for
weight reuse. Reward metadata distinguishes their semantics and prevents mixing
Replay. Smooth mode exposes headroom ratios in [0, 2], initially both 1.
"""

import numpy as np

from pvz_gym.observation import (OBSERVATION_SCHEMA, OBSERVATION_SPEC,
                                LEGACY_OBSERVATION_SCHEMA, LEGACY_OBSERVATION_SPEC)

DENSE_OBSERVATION_SCHEMA = OBSERVATION_SCHEMA + "_transition_dense_context_v1"
LEGACY_DENSE_OBSERVATION_SCHEMA = LEGACY_OBSERVATION_SCHEMA + "_transition_dense_context_v1"


def observation_schema(reward_context=False, boss_projectiles=True):
    base = OBSERVATION_SCHEMA if boss_projectiles else LEGACY_OBSERVATION_SCHEMA
    return base + "_transition_dense_context_v1" if reward_context else base


def observation_spec(reward_context=False, boss_projectiles=True):
    spec = OBSERVATION_SPEC if boss_projectiles else LEGACY_OBSERVATION_SPEC
    if not reward_context:
        return spec
    return {**spec, "reward_context": ((3,), np.dtype(np.float32))}
