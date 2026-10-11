"""按动作家族均衡探索 PvZ 扁平动作空间。"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from pvz_gym.action_space import (
    ACTION_SPACE_SIZE,
    COB_BASE,
    COLS,
    INTERACT_BASE,
    MAX_ROWS,
    PLANT_BASE,
    SHOVEL_NORMAL_BASE,
)


ACTION_FAMILY_NAMES = ("noop", "plant", "shovel", "cob", "interact")
ACTION_FAMILY_RANGES = (
    (0, PLANT_BASE),
    (PLANT_BASE, SHOVEL_NORMAL_BASE),
    (SHOVEL_NORMAL_BASE, COB_BASE),
    (COB_BASE, INTERACT_BASE),
    (INTERACT_BASE, ACTION_SPACE_SIZE),
)
UNIFORM_ACTION_FAMILY_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0)
EXPLORATION_MODE_LABELS = {
    "q_guided_mixed": "贪心＋Q 引导＋广泛",
    "epsilon_random": "贪心＋纯随机",
    "q_guided_neighbor": "贪心＋Q 引导＋邻格",
}
DEFAULT_EXPLORATION_MODE = "q_guided_mixed"
DEFAULT_EXPLORATION_RATIO = (50.0, 50.0)


def q_guided_fraction(mode: str, ratio: Sequence[float]) -> float:
    """Direct Q:random weights within exploration; epsilon stays independent."""
    if mode not in EXPLORATION_MODE_LABELS:
        raise ValueError(f"unsupported exploration_mode: {mode!r}")
    if len(ratio) != 2 or any(not math.isfinite(value) or value < 0 for value in ratio):
        raise ValueError("exploration_ratio requires two finite non-negative weights")
    scale = max(ratio)
    if scale == 0:
        raise ValueError("exploration_ratio cannot be 0:0")
    if mode == "epsilon_random":
        return 0.0
    q_weight, random_weight = (value / scale for value in ratio)
    return q_weight / (q_weight + random_weight)


def parse_exploration_ratio(raw: str) -> tuple[float, float]:
    """Parse Q:random, e.g. 50:50 or 80:20, with zero-weight branches allowed."""
    try:
        ratio = tuple(float(item.strip()) for item in str(raw).split(":"))
    except ValueError as error:
        raise ValueError("探索比例须填写 Q:随机，例如 50:50") from error
    q_guided_fraction(DEFAULT_EXPLORATION_MODE, ratio)
    return ratio


def format_exploration_ratio(ratio: Sequence[float]) -> str:
    return ":".join(format(value, ".17g") for value in ratio)


def _validated_family_weights(type_weights: Sequence[float]) -> np.ndarray:
    weights = np.asarray(tuple(type_weights), dtype=np.float64)
    if weights.shape != (len(ACTION_FAMILY_RANGES),):
        raise ValueError("type_weights must contain noop, plant, shovel, cob, and interact weights")
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError("type_weights must be finite and non-negative")
    return weights


def sample_q_guided_action(
    q_values,
    action_mask,
    rng: np.random.Generator,
    type_weights: Sequence[float] = UNIFORM_ACTION_FAMILY_WEIGHTS,
) -> int:
    """Choose a legal family by weight, then an action by within-family Q.

    Family probabilities are independent of action counts and Q values.
    Within-family standardization preserves reward-scale invariance. If all
    available family weights are zero, use equal family probabilities.
    """
    mask = np.asarray(action_mask, dtype=np.bool_)
    values = np.asarray(q_values, dtype=np.float64)
    if mask.shape != (ACTION_SPACE_SIZE,) or values.shape != mask.shape:
        raise ValueError("Q guidance requires matching full action vectors")

    legal = np.flatnonzero(mask)
    if not legal.size:
        raise RuntimeError("PLAYING state has no legal actions")
    weights = _validated_family_weights(type_weights)

    families = np.searchsorted([end for _, end in ACTION_FAMILY_RANGES], legal, side="right")
    scores = values[legal]
    if not np.isfinite(scores).all():
        raise ValueError("legal Q values must be finite")
    available = np.unique(families)
    family_probabilities = weights[available].copy()
    max_weight = family_probabilities.max()
    if max_weight > 0:
        family_probabilities /= max_weight
    else:
        family_probabilities.fill(1.0)
    family_probabilities /= family_probabilities.sum()

    # Combine the two stages into one draw with the same family marginals.
    probabilities = np.zeros(legal.size, dtype=np.float64)
    for family, family_probability in zip(available, family_probabilities):
        if family_probability == 0:
            continue
        members = families == family
        family_scores = scores[members]
        scale = float(np.std(family_scores)) or 1.0
        logits = (family_scores - family_scores.max()) / scale
        conditional = np.exp(logits)
        probabilities[members] = family_probability * conditional / conditional.sum()
    probabilities /= probabilities.sum()
    return int(rng.choice(legal, p=probabilities))


def legal_neighbor_actions(anchor: int, action_mask) -> np.ndarray:
    """Eight surrounding cells, preserving card slot and action subtype."""
    mask = np.asarray(action_mask, dtype=np.bool_)
    if mask.shape != (ACTION_SPACE_SIZE,):
        raise ValueError("neighbor exploration requires a full action mask")
    if not 0 <= anchor < ACTION_SPACE_SIZE:
        raise ValueError("anchor action out of range")
    # 铲子及非种植动作禁用近格探索，只有种植具备空间连续性
    if not (PLANT_BASE <= anchor < SHOVEL_NORMAL_BASE):
        return np.empty(0, dtype=np.int64)
    start = PLANT_BASE
    cells = MAX_ROWS * COLS
    group_start = start + ((anchor - start) // cells) * cells
    row, col = divmod(anchor - group_start, COLS)
    neighbors = [
        group_start + nr * COLS + nc
        for nr in range(max(0, row - 1), min(MAX_ROWS, row + 2))
        for nc in range(max(0, col - 1), min(COLS, col + 2))
        if (nr != row or nc != col) and mask[group_start + nr * COLS + nc]
    ]
    return np.asarray(neighbors, dtype=np.int64)


def sample_neighbor_action(
    neighbors: Sequence[int] | np.ndarray,
    q_values,
    rng: np.random.Generator,
) -> int:
    """按 Q 值分布从合法邻格动作中采样，修复纯随机忽略 Q 的问题。"""
    actions = np.asarray(neighbors, dtype=np.int64)
    if not actions.size:
        raise ValueError("neighbors cannot be empty")
    scores = np.asarray(q_values, dtype=np.float64)[actions]
    if not np.isfinite(scores).all():
        raise ValueError("neighbor Q values must be finite")
    scale = float(np.std(scores)) or 1.0
    logits = (scores - scores.max()) / scale
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    return int(rng.choice(actions, p=probabilities))


def parse_action_family_weights(raw: str) -> tuple[float, float, float, float, float]:
    """解析空操作、种植、铲除、玉米炮、格子交互五类探索权重。"""
    try:
        values = tuple(float(item.strip()) for item in str(raw).split(",") if item.strip())
    except ValueError as error:
        raise ValueError("exploration type weights must be numeric") from error
    if len(values) != 5:
        raise ValueError("exploration type weights require five comma-separated values")
    return values


def sample_balanced_legal_action(
    action_mask,
    rng: np.random.Generator,
    type_weights: Sequence[float] = UNIFORM_ACTION_FAMILY_WEIGHTS,
) -> int:
    """在合法动作中分层采样，避免动作数量多的家族天然占据更高探索概率。

    先按配置权重选择当前至少有一个合法动作的动作家族，再在该家族内均匀采样。
    权重为零的家族在存在其他正权重家族时会被跳过；如果当前可用家族权重全为零，
    则退回等权重选择，保证始终能返回一个合法动作。
    """

    mask = np.asarray(action_mask, dtype=np.bool_)
    if mask.shape != (ACTION_SPACE_SIZE,):
        raise ValueError(
            f"balanced exploration requires mask shape {(ACTION_SPACE_SIZE,)}, "
            f"got {mask.shape}"
        )
    weights = _validated_family_weights(type_weights)

    available_actions: list[np.ndarray] = []
    available_weights: list[float] = []
    for weight, (start, end) in zip(weights, ACTION_FAMILY_RANGES):
        legal = np.flatnonzero(mask[start:end]) + start
        if legal.size:
            available_actions.append(legal)
            available_weights.append(float(weight))

    if not available_actions:
        raise RuntimeError("PLAYING state has no legal actions")

    probabilities = np.asarray(available_weights, dtype=np.float64)
    if probabilities.sum() <= 0.0:
        probabilities.fill(1.0)
    probabilities /= probabilities.sum()
    family_index = int(rng.choice(len(available_actions), p=probabilities))
    return int(rng.choice(available_actions[family_index]))


__all__ = [
    "EXPLORATION_MODE_LABELS",
    "DEFAULT_EXPLORATION_MODE",
    "DEFAULT_EXPLORATION_RATIO",
    "parse_exploration_ratio",
    "format_exploration_ratio",
    "q_guided_fraction",
    "sample_q_guided_action",
    "legal_neighbor_actions",
    "sample_neighbor_action",
    "ACTION_FAMILY_NAMES",
    "ACTION_FAMILY_RANGES",
    "UNIFORM_ACTION_FAMILY_WEIGHTS",
    "parse_action_family_weights",
    "sample_balanced_legal_action",
]
