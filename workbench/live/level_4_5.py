"""Adventure 4-5 Vasebreaker (砸罐子) LIVE entry points."""

from __future__ import annotations

from typing import Any

from agents.scripted.level_4_5 import choose_level_4_5_plant

from . import early_levels_live_tests as _early


LEVEL_4_5_SPEC = _early.EarlyLevelSpec(
    area=4,
    sub_level=5,
    level_id=35,
    label="4-5",
    adventure1_priority=(),
    adventure2_priority=(),
    adventure1_policy=choose_level_4_5_plant,
    adventure2_policy=choose_level_4_5_plant,
    max_plants=240,
)


def run_level_4_5_e2e(client: Any, **kwargs) -> _early.EarlyLevelE2EResult:
    kwargs.setdefault("max_ticks", 150000)
    return _early.run_early_level_e2e(client, LEVEL_4_5_SPEC, **kwargs)


def run_level_4_5_adventure1_e2e(client: Any, **kwargs) -> _early.EarlyLevelE2EResult:
    return run_level_4_5_e2e(client, expected_path=_early.PLAYTHROUGH_1, **kwargs)


def run_level_4_5_adventure2_e2e(client: Any, **kwargs) -> _early.EarlyLevelE2EResult:
    return run_level_4_5_e2e(client, expected_path=_early.PLAYTHROUGH_2, **kwargs)


def run_level_4_5_stress(client: Any, **kwargs) -> _early.EarlyLevelStressResult:
    kwargs.setdefault("max_ticks", 150000)
    return _early.run_early_level_stress(client, LEVEL_4_5_SPEC, **kwargs)
