"""Validation helpers for structured PvZ observations.

The environment already returns model-ready arrays from ``encode_observation``.
Production training must not copy and sanitize the full observation a second time.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from pvz_gym.observation import OBSERVATION_SPEC


def validate_observation(observation: Mapping[str, np.ndarray]) -> None:
    """Validate the structured contract without copying or mutating any field."""
    if not isinstance(observation, Mapping) or set(observation) != set(OBSERVATION_SPEC):
        raise ValueError("observation does not match the structured Observation contract")
    for name, (shape, _dtype) in OBSERVATION_SPEC.items():
        value = np.asarray(observation[name])
        if value.shape != shape:
            raise ValueError(f"unexpected {name} shape: {value.shape} != {shape}")


def normalize_observation(observation: Mapping[str, np.ndarray], *,
                          boss_projectiles: bool = True) -> Mapping[str, np.ndarray]:
    """Compatibility no-op for already encoded observations.

    ``pvz_gym.observation.encode_observation`` owns numeric cleanup and clipping.
    Keeping this function as a no-op avoids breaking existing callers while
    removing the old 19-field copy + nan_to_num + clip pass from every reset/step.
    Use :func:`validate_observation` explicitly in debug/tests when desired.
    """
    if not boss_projectiles:
        raise ValueError("Boss projectile input is required; legacy observation projection is unsupported")
    return observation


__all__ = ["normalize_observation", "validate_observation"]
