# -*- coding: utf-8 -*-
"""Registry for all 5x10 Adventure LIVE environment tests.

1-1..1-5 and 1-10 keep level-specific policies. Remaining levels use a
deliberately simple "first legal Plant" baseline so the LIVE suite can exercise
lifecycle, masks, actions and terminal detection across all 50 Adventure levels
immediately. Special levels that need a richer action domain should fail visibly
instead of being silently skipped; that makes the 50-level suite useful as a
coverage map.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .early_levels_live_tests import EarlyLevelSpec, LEVEL_1_2_SPEC, LEVEL_1_3_SPEC
from .level_1_1_playthrough_live_tests import LEVEL_1_1_SPEC
from .level_1_4_live_tests import LEVEL_1_4_SPEC
from .level_1_5 import LEVEL_1_5_SPEC
from .level_1_10 import LEVEL_1_10_SPEC
from .level_2_10 import LEVEL_2_10_SPEC
from .level_4_5 import LEVEL_4_5_SPEC


SCENES = (
    (1, "白天草地", "DAY"),
    (2, "夜晚", "NIGHT"),
    (3, "泳池", "POOL_DAY"),
    (4, "雾夜", "FOG"),
    (5, "屋顶", "ROOF"),
)

ACTION_SPACE_SIZE = 757
PLANT_BASE = 1
MAX_CARD_SLOTS = 10
MAX_ROWS = 6
COLS = 9
from agents.scripted.generic_policy import (
    ENHANCED_GENERIC_DECK_PRIORITY,
    choose_enhanced_generic_plant,
)
from pvz_gym.default_decks import default_deck_priority

GENERIC_DECK_PRIORITY = ENHANCED_GENERIC_DECK_PRIORITY
choose_generic_legal_plant = choose_enhanced_generic_plant


def _generic_spec(area: int, sub_level: int) -> EarlyLevelSpec:
    level_id = (area - 1) * 10 + sub_level
    return EarlyLevelSpec(
        area=area,
        sub_level=sub_level,
        level_id=level_id,
        label=f"{area}-{sub_level}",
        adventure1_priority=default_deck_priority(level_id, 1),
        adventure2_priority=default_deck_priority(level_id, 2),
        adventure1_policy=choose_generic_legal_plant,
        adventure2_policy=choose_generic_legal_plant,
        max_plants=120,
    )


_SPECIALIZED = {
    1: LEVEL_1_1_SPEC,
    2: LEVEL_1_2_SPEC,
    3: LEVEL_1_3_SPEC,
    4: LEVEL_1_4_SPEC,
    5: LEVEL_1_5_SPEC,
    10: LEVEL_1_10_SPEC,
    20: LEVEL_2_10_SPEC,
    35: LEVEL_4_5_SPEC,
}

LEVEL_SPECS: Dict[int, EarlyLevelSpec] = {}
for _area in range(1, 6):
    for _sub in range(1, 11):
        _level_id = (_area - 1) * 10 + _sub
        LEVEL_SPECS[_level_id] = _SPECIALIZED.get(
            _level_id, _generic_spec(_area, _sub)
        )


def get_level_spec(level_id: int) -> EarlyLevelSpec:
    level_id = int(level_id)
    if level_id not in LEVEL_SPECS:
        raise ValueError(f"level_id must be 1..50, got {level_id}")
    return LEVEL_SPECS[level_id]


def get_scene_specs(area: int):
    area = int(area)
    if area < 1 or area > 5:
        raise ValueError(f"area must be 1..5, got {area}")
    start = (area - 1) * 10 + 1
    return [LEVEL_SPECS[i] for i in range(start, start + 10)]


def all_level_specs():
    return [LEVEL_SPECS[i] for i in range(1, 51)]
