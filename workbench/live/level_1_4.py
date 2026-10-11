# -*- coding: utf-8 -*-
"""Adventure 1-4 LIVE/Gym wrappers using the tuned 1-4 scripted policy."""

from __future__ import annotations

from typing import Any

from . import early_levels_live_tests as _early
from .level_1_4_policy import (
    LEVEL_1_4_FIRST_RUN_DECK_PRIORITY,
    LEVEL_1_4_REPLAY_DECK_PRIORITY,
    choose_level_1_4_first_run_plant,
    choose_level_1_4_replay_plant,
)


LEVEL_1_4_SPEC = _early.EarlyLevelSpec(
    area=1,
    sub_level=4,
    level_id=4,
    label="1-4",
    adventure1_priority=tuple(LEVEL_1_4_FIRST_RUN_DECK_PRIORITY),
    adventure2_priority=tuple(LEVEL_1_4_REPLAY_DECK_PRIORITY),
    adventure1_policy=choose_level_1_4_first_run_plant,
    adventure2_policy=choose_level_1_4_replay_plant,
    max_plants=60,
)


def _e2e_defaults(kwargs: dict) -> dict:
    values = dict(kwargs)
    # Give the 10-wave level a wider safety budget than the tutorial-like 1-1.
    values.setdefault("max_ticks", 90000)
    values.setdefault("step_ticks", 50)
    values.setdefault("max_plants", LEVEL_1_4_SPEC.max_plants)
    return values


def _stress_defaults(kwargs: dict) -> dict:
    values = dict(kwargs)
    values.setdefault("max_ticks", 90000)
    values.setdefault("step_ticks", 50)
    return values


def run_level_1_4_e2e(client: Any, **kwargs) -> _early.EarlyLevelE2EResult:
    return _early.run_early_level_e2e(
        client,
        LEVEL_1_4_SPEC,
        **_e2e_defaults(kwargs),
    )


def run_level_1_4_adventure1_e2e(
    client: Any,
    **kwargs,
) -> _early.EarlyLevelE2EResult:
    return _early.run_early_level_e2e(
        client,
        LEVEL_1_4_SPEC,
        expected_path=_early.PLAYTHROUGH_1,
        **_e2e_defaults(kwargs),
    )


def run_level_1_4_adventure2_e2e(
    client: Any,
    **kwargs,
) -> _early.EarlyLevelE2EResult:
    return _early.run_early_level_e2e(
        client,
        LEVEL_1_4_SPEC,
        expected_path=_early.PLAYTHROUGH_2,
        **_e2e_defaults(kwargs),
    )


def run_level_1_4_adventure1_stress(
    client: Any,
    **kwargs,
) -> _early.EarlyLevelStressResult:
    return _early.run_early_level_stress(
        client,
        LEVEL_1_4_SPEC,
        expected_path=_early.PLAYTHROUGH_1,
        **_stress_defaults(kwargs),
    )


def run_level_1_4_adventure2_stress(
    client: Any,
    **kwargs,
) -> _early.EarlyLevelStressResult:
    return _early.run_early_level_stress(
        client,
        LEVEL_1_4_SPEC,
        expected_path=_early.PLAYTHROUGH_2,
        **_stress_defaults(kwargs),
    )


# Compatibility aliases used by the older first-run/replay naming.
run_level_1_4_first_run_e2e = run_level_1_4_adventure1_e2e
run_level_1_4_replay_e2e = run_level_1_4_adventure2_e2e
run_level_1_4_stress = run_level_1_4_adventure2_stress
