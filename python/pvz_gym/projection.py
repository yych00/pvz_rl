"""把原生事实转换成直接交给智能体的可变长观测表示。"""

from __future__ import annotations

from .types import Observation


def project_observation(observation: Observation) -> dict:
    # Gymnasium 的 Sequence 需要使用元组形式表示可变长度序列。
    return {
        "global": dict(observation.global_state),
        "cards": tuple(dict(item) for item in observation.cards),
        "plants": tuple(dict(item) for item in observation.plants),
        "zombies": tuple(dict(item) for item in observation.zombies),
        "dropped_cards": tuple(dict(item) for item in observation.dropped_cards),
        "grid_items": tuple(dict(item) for item in observation.grid_items),
        "ice_trails": tuple(dict(item) for item in observation.ice_trails),
        "mowers": tuple(dict(item) for item in observation.mowers),
        "grid": tuple(dict(item) for item in observation.grid),
    }
