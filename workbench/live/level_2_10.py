"""Adventure 2-10 night conveyor LIVE entry points."""

from __future__ import annotations

from typing import Any

from agents.scripted.level_2_10 import choose_level_2_10_plant

from . import early_levels_live_tests as _early


LEVEL_2_10_SPEC = _early.EarlyLevelSpec(
    area=2,
    sub_level=10,
    level_id=20,
    label="2-10",
    adventure1_priority=(),
    adventure2_priority=(),
    adventure1_policy=choose_level_2_10_plant,
    adventure2_policy=choose_level_2_10_plant,
    max_plants=200,
)


def run_level_2_10_e2e(client: Any, **kwargs) -> _early.EarlyLevelE2EResult:
    kwargs.setdefault("max_ticks", 140000)
    return _early.run_early_level_e2e(client, LEVEL_2_10_SPEC, **kwargs)


def run_level_2_10_adventure1_e2e(client: Any, **kwargs) -> _early.EarlyLevelE2EResult:
    return run_level_2_10_e2e(client, expected_path=_early.PLAYTHROUGH_1, **kwargs)


def run_level_2_10_adventure2_e2e(client: Any, **kwargs) -> _early.EarlyLevelE2EResult:
    return run_level_2_10_e2e(client, expected_path=_early.PLAYTHROUGH_2, **kwargs)


def run_level_2_10_stress(client: Any, **kwargs) -> _early.EarlyLevelStressResult:
    kwargs.setdefault("max_ticks", 140000)
    return _early.run_early_level_stress(client, LEVEL_2_10_SPEC, **kwargs)
