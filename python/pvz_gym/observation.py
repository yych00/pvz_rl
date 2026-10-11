"""把原生游戏事实投影成固定形状的 Gym 观测。

实例 ID 只保留在原始 Observation 中用于跨游戏刻追踪，不作为模型输入。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .action_space import COLS, MAX_CARD_SLOTS, MAX_ROWS

LEGACY_OBSERVATION_SCHEMA = "pvz_observation_entities_256_zombie_grid_imitater_boss_nonspatial_v5"
PREVIOUS_OBSERVATION_SCHEMA = "pvz_observation_entities_256_zombie_grid_imitater_boss_projectiles_v4"
OBSERVATION_SCHEMA = "pvz_observation_entities_256_zombie_grid_imitater_boss_projectiles_boss_nonspatial_v5"
ZOMBIE_BOSS_TYPE = 25
ZOMBIE_BOSS_CATEGORY = ZOMBIE_BOSS_TYPE + 1
PLANT_LAYERS = 3  # 同格多层植物，例如底座、主体、南瓜。
MAX_ZOMBIES = 256  # 无尽后期保留更多僵尸个体，避免过早聚合。
MAX_DROPPED_CARDS = 16
MAX_BOSS_PROJECTILES = 8
BOSS_PROJECTILE_FIELDS = ("x_norm", "y_norm")
BOSS_PROJECTILE_FEATURES = len(BOSS_PROJECTILE_FIELDS)
TYPE_CAPACITY = 256
STATE_CAPACITY = 256
SCENE_CAPACITY = 16
ZOMBIE_ROSTER_SIZE = 64

GLOBAL_FEATURES = 8
CARD_FEATURES = 6
PLANT_FEATURES = 4
GRID_FEATURES = 9
ZOMBIE_GRID_FEATURES = 2  # count, total effective HP
# 连续事实与类别字段分开；类型、阶段、高度、行和目标位置交给嵌入层处理。
ZOMBIE_CONTINUOUS_FIELDS = (
    "x_norm", "y_norm", "vel_x", "altitude", "body_hp", "max_hp",
    "helm_hp", "shield_hp", "flying_hp", "phase_counter",
    "chilled_counter", "buttered_counter", "ice_trap_counter",
    "is_eating", "is_mind_controlled",
)
ZOMBIE_FEATURES = len(ZOMBIE_CONTINUOUS_FIELDS)
DROPPED_CARD_FEATURES = 4
MOWER_STATES = 4

# 回放缓存、Gym 空间和 DQN 共用的固定字段契约。
OBSERVATION_SPEC: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "global": ((GLOBAL_FEATURES,), np.dtype(np.float32)),
    "scene": ((1,), np.dtype(np.int64)),
    "zombie_roster": ((ZOMBIE_ROSTER_SIZE,), np.dtype(np.int8)),
    "card_type": ((MAX_CARD_SLOTS,), np.dtype(np.int64)),
    "cards": ((MAX_CARD_SLOTS, CARD_FEATURES), np.dtype(np.float32)),
    "plant_type": ((PLANT_LAYERS, MAX_ROWS, COLS), np.dtype(np.int64)),
    "plant_state": ((PLANT_LAYERS, MAX_ROWS, COLS), np.dtype(np.int64)),
    "plants": ((PLANT_LAYERS, MAX_ROWS, COLS, PLANT_FEATURES), np.dtype(np.float32)),
    "grid_terrain": ((MAX_ROWS, COLS), np.dtype(np.int64)),
    "grid": ((MAX_ROWS, COLS, GRID_FEATURES), np.dtype(np.float32)),
    "zombie_grid": ((MAX_ROWS, COLS, ZOMBIE_GRID_FEATURES), np.dtype(np.float32)),
    "zombie_type": ((MAX_ZOMBIES,), np.dtype(np.int64)),
    "zombie_phase": ((MAX_ZOMBIES,), np.dtype(np.int64)),
    "zombie_height": ((MAX_ZOMBIES,), np.dtype(np.int64)),
    "zombie_target_row": ((MAX_ZOMBIES,), np.dtype(np.int64)),
    "zombie_target_col": ((MAX_ZOMBIES,), np.dtype(np.int64)),
    "zombie_row": ((MAX_ZOMBIES,), np.dtype(np.int64)),
    "zombies": ((MAX_ZOMBIES, ZOMBIE_FEATURES), np.dtype(np.float32)),
    "zombie_mask": ((MAX_ZOMBIES,), np.dtype(np.int8)),
    "dropped_card_type": ((MAX_DROPPED_CARDS,), np.dtype(np.int64)),
    "dropped_cards": ((MAX_DROPPED_CARDS, DROPPED_CARD_FEATURES), np.dtype(np.float32)),
    "mower_state": ((MAX_ROWS,), np.dtype(np.int64)),
}
LEGACY_OBSERVATION_SPEC = dict(OBSERVATION_SPEC)
# Append projectile fields; both schemas share the nonspatial Boss contract.
OBSERVATION_SPEC.update({
    "boss_projectile_type": ((MAX_BOSS_PROJECTILES,), np.dtype(np.int64)),
    "boss_projectile_row": ((MAX_BOSS_PROJECTILES,), np.dtype(np.int64)),
    "boss_projectiles": ((MAX_BOSS_PROJECTILES, BOSS_PROJECTILE_FEATURES), np.dtype(np.float32)),
    "boss_projectile_mask": ((MAX_BOSS_PROJECTILES,), np.dtype(np.int8)),
})


def _field(value: object, name: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if np.isfinite(result) else default


def _category(value: Any, capacity: int = TYPE_CAPACITY) -> int:
    """把原生的 -1/从 0 开始枚举映射成 0 填充、从 1 开始的嵌入索引。"""
    result = int(_number(value, -1))
    return min(max(result + 1, 0), capacity)


def _log_scale(value: Any, reference: float) -> float:
    return float(np.clip(np.log1p(max(0.0, _number(value))) / np.log1p(reference), 0.0, 4.0))


def _signed_log_scale(value: Any, reference: float) -> float:
    number = _number(value)
    return float(np.sign(number)) * _log_scale(abs(number), reference)


def zombie_truncation_info(raw_obs: object) -> dict[str, int | bool]:
    """仅用于诊断 256 上限是否发生截断，不进入策略观测。"""
    dropped = max(0, len(_field(raw_obs, "zombies", ()) or ()) - MAX_ZOMBIES)
    return {"zombie_truncated": dropped > 0, "zombies_dropped": dropped}


def boss_projectile_truncation_info(raw_obs: object) -> dict[str, int | bool]:
    dropped = max(0, len(_valid_boss_projectiles(raw_obs)) - MAX_BOSS_PROJECTILES)
    return {"boss_projectiles_truncated": dropped > 0, "boss_projectiles_dropped": dropped}


def _valid_boss_projectiles(raw_obs: object) -> list[tuple[int, int, float, float]]:
    entities = []
    for entity in _field(raw_obs, "boss_projectiles", ()) or ():
        if not bool(_field(entity, "valid", False)):
            continue
        try:
            kind = int(_field(entity, "type", -1))
            row = int(_field(entity, "row", -1))
            x, y = float(_field(entity, "x", float("nan"))), float(_field(entity, "y", float("nan")))
        except (TypeError, ValueError, OverflowError):
            continue
        if kind in (0, 1) and 0 <= row < MAX_ROWS and np.isfinite(x) and np.isfinite(y):
            entities.append((kind, row, x, y))
    # Normally one Boss owns at most one ball. For injected multiple Bosses,
    # retain the eight leftmost valid attacks, report overflow independently.
    return sorted(entities, key=lambda value: (value[2], value[1], value[0]))


def _ratio(value: Any, denominator: Any) -> float:
    den = max(0.0, _number(denominator))
    if den <= 0.0:
        return 0.0
    return float(np.clip(max(0.0, _number(value)) / den, 0.0, 4.0))


def _zombie_effective_hp(zombie: object) -> float:
    """用于格子汇总的总有效耐久；不加入类型或威胁权重。"""
    body = max(0.0, _number(_field(zombie, "body_hp", _field(zombie, "hp", 0))))
    return body + sum(
        max(0.0, _number(_field(zombie, name, 0)))
        for name in ("helm_hp", "shield_hp", "flying_hp")
    )


def _zombie_grid_col(zombie: object) -> int:
    """按 Board 的 LAWN_XMIN=40、80px 网格投影到 0..8 列；越界出生点夹到边缘格。"""
    x = _number(_field(zombie, "x", 0))
    return int(np.clip(np.floor((x - 40.0) / 80.0), 0, COLS - 1))


def _fog_visibility(global_state: object):
    """Sample native fog cells, not inferred lamp radii or action timers.

    This is a cell-level visibility contract, not sprite/pixel segmentation.
    Alpha >=128 is obscured. The native offset follows blowing/returning fog.
    The rightmost sample repeats beyond the board as in Board::DrawFog.
    """
    if int(_number(_field(global_state, "scene", -1), -1)) != 3:
        return lambda row, x: True
    if not bool(_field(global_state, "fog_state_available", False)):
        raise ValueError("fog visibility requires a rebuilt native DLL with fog state")
    alpha = np.asarray(_field(global_state, "fog_alpha", ()), dtype=np.float64)
    offset = float(_field(global_state, "fog_offset", float("nan")))
    if (alpha.shape != (COLS * (MAX_ROWS + 1),) or not np.isfinite(alpha).all()
            or (alpha < 0).any() or (alpha > 255).any()
            or not np.isfinite(offset) or not 0 <= offset <= 2000):
        raise ValueError("invalid native fog state")

    def visible(row, x):
        if not 0 <= row < MAX_ROWS or not np.isfinite(x):
            return False
        col = int(np.floor((x - 40.0 - offset) / 80.0))
        return col < 0 or alpha[min(col, COLS - 1) * (MAX_ROWS + 1) + row] < 128

    return visible


def empty_observation(*, boss_projectiles: bool = True) -> dict[str, np.ndarray]:
    spec = OBSERVATION_SPEC if boss_projectiles else LEGACY_OBSERVATION_SPEC
    return {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in spec.items()}


def encode_observation(raw_obs: object, *, boss_projectiles: bool = True) -> dict[str, np.ndarray]:
    """把原始 PvZ 事实编码成有限、固定形状的结构化观测。"""
    result = empty_observation(boss_projectiles=boss_projectiles)
    global_state = _field(raw_obs, "global_state", None)
    if global_state is None:
        global_state = _field(raw_obs, "global", {})
    visible = _fog_visibility(global_state)

    sun = _number(_field(global_state, "sun", 0))
    current_wave = max(0.0, _number(_field(global_state, "current_wave", 0)))
    total_waves = max(0.0, _number(_field(global_state, "total_waves", 0)))
    countdown = max(0.0, _number(_field(global_state, "wave_countdown", 0)))
    active_rows = min(max(_number(_field(global_state, "grid_rows", MAX_ROWS)), 0.0), MAX_ROWS)
    result["global"][:] = (
        _log_scale(sun, 9990.0),
        _log_scale(current_wave, 100.0),
        _log_scale(total_waves, 100.0),
        float(np.clip(current_wave / max(total_waves, 1.0), 0.0, 2.0)),
        _log_scale(countdown, 10000.0),
        1.0 if bool(_field(global_state, "is_huge_wave", False)) else 0.0,
        1.0 if bool(_field(global_state, "is_paused", False)) else 0.0,
        active_rows / MAX_ROWS,
    )
    result["scene"][0] = _category(_field(global_state, "scene", -1), SCENE_CAPACITY - 1)
    for zombie_type in _field(global_state, "zombie_types_in_level", ()) or ():
        index = int(_number(zombie_type, -1))
        if 0 <= index < ZOMBIE_ROSTER_SIZE:
            result["zombie_roster"][index] = 1

    # 类型保留模仿目标；独立标记让动作评分区分普通卡和模仿者卡。
    for card in _field(raw_obs, "cards", ()) or ():
        slot = int(_number(_field(card, "slot", -1), -1))
        if not 0 <= slot < MAX_CARD_SLOTS:
            continue
        duration = max(0.0, _number(_field(card, "cooldown_duration", 0)))
        remaining = max(0.0, _number(_field(card, "cooldown_remaining", 0)))
        result["card_type"][slot] = _category(_field(card, "plant_type", -1))
        result["cards"][slot] = (
            1.0,
            _log_scale(_field(card, "cost", 0), 500.0),
            float(np.clip(1.0 - remaining / max(duration, 1.0), 0.0, 1.0)),
            float(np.clip(remaining / max(duration, 1.0), 0.0, 1.0)),
            1.0 if bool(_field(card, "ready", False)) else 0.0,
            1.0 if bool(_field(card, "is_imitater", False)) else 0.0,
        )

    # 自己的植物不受迷雾过滤；三层避免底座、主体、南瓜信息互相覆盖。
    for plant in _field(raw_obs, "plants", ()) or ():
        row = int(_number(_field(plant, "row", -1), -1))
        col = int(_number(_field(plant, "col", -1), -1))
        layer = int(_number(_field(plant, "layer", -1), -1))
        if not (0 <= row < MAX_ROWS and 0 <= col < COLS and 0 <= layer < PLANT_LAYERS):
            continue
        hp = max(0.0, _number(_field(plant, "hp", 0)))
        max_hp = max(0.0, _number(_field(plant, "max_hp", 0)))
        result["plant_type"][layer, row, col] = _category(_field(plant, "type", -1))
        result["plant_state"][layer, row, col] = _category(
            _field(plant, "state", -1), STATE_CAPACITY - 1
        )
        result["plants"][layer, row, col] = (
            1.0,
            _ratio(hp, max_hp),
            _log_scale(hp, 10000.0),
            1.0 if bool(_field(plant, "is_sleeping", False)) else 0.0,
        )

    # 棋盘网格描述地形、占用、冰道和格子物件；僵尸另有两个简单空间通道。
    for square in _field(raw_obs, "grid", ()) or ():
        row = int(_number(_field(square, "row", -1), -1))
        col = int(_number(_field(square, "col", -1), -1))
        if not (0 <= row < MAX_ROWS and 0 <= col < COLS):
            continue
        result["grid_terrain"][row, col] = _category(_field(square, "terrain", -1))
        # 植物占用与植物实体一致，迷雾中仍保留；危险地形继续按可见性过滤。
        result["grid"][row, col, :4] = (
            1.0,
            bool(_field(square, "has_base_plant", False)),
            bool(_field(square, "has_main_plant", False)),
            bool(_field(square, "has_pumpkin", False)),
        )
        if not visible(row, 80.0 + 80.0 * col):
            continue
        result["grid"][row, col, 4:6] = (
            bool(_field(square, "has_grave", False)),
            bool(_field(square, "has_crater", False)),
        )

    for trail in _field(raw_obs, "ice_trails", ()) or ():
        row = int(_number(_field(trail, "row", -1), -1))
        start = int(_number(_field(trail, "col_start", -1), -1))
        end = int(_number(_field(trail, "col_end", -1), -1))
        if not 0 <= row < MAX_ROWS:
            continue
        for col in range(max(0, min(start, end)), min(COLS - 1, max(start, end)) + 1):
            if not visible(row, 80.0 + 80.0 * col):
                continue
            result["grid"][row, col, 6] = max(
                result["grid"][row, col, 6],
                _log_scale(_field(trail, "melt_countdown", 0), 10000.0),
            )

    for item in _field(raw_obs, "grid_items", ()) or ():
        row = int(_number(_field(item, "row", -1), -1))
        col = int(_number(_field(item, "col", -1), -1))
        if (0 <= row < MAX_ROWS and 0 <= col < COLS
                and visible(row, 80.0 + 80.0 * col)):
            result["grid"][row, col, 7] += 1.0
            result["grid"][row, col, 8] = max(
                result["grid"][row, col, 8],
                _log_scale(_field(item, "remaining_ticks", 0), 10000.0),
            )
    result["grid"][:, :, 7] = np.log1p(result["grid"][:, :, 7]) / np.log(5.0)

    # 僵尸仍保留为独立实体；除无本体空间位置的僵王外，额外投影两个 6×9 空间统计：
    # [0] 格内数量，[1] 格内 body + helm + shield + flying 总 HP。
    zombies = [z for z in (_field(raw_obs, "zombies", ()) or ())
               if visible(int(_number(_field(z, "row", -1), -1)),
                          _number(_field(z, "x", 0)))]
    for zombie in zombies:
        # The Boss's base x/y/row are placeholders, not a lawn location.
        if int(_number(_field(zombie, "type", -1), -1)) == ZOMBIE_BOSS_TYPE:
            continue
        row = int(_number(_field(zombie, "row", -1), -1))
        if not 0 <= row < MAX_ROWS:
            continue
        col = _zombie_grid_col(zombie)
        result["zombie_grid"][row, col, 0] += 1.0
        result["zombie_grid"][row, col, 1] += _zombie_effective_hp(zombie)

    # 数量与总 HP 用固定 log 尺度归一化，避免无尽高密度场面把 CNN 输入冲爆。
    result["zombie_grid"][:, :, 0] = np.clip(
        np.log1p(result["zombie_grid"][:, :, 0]) / np.log1p(4.0), 0.0, 4.0
    )
    result["zombie_grid"][:, :, 1] = np.clip(
        np.log1p(result["zombie_grid"][:, :, 1]) / np.log1p(20000.0), 0.0, 4.0
    )

    # 行为阶段、高度状态和计时器继续通过实体路径保留矿工、小丑、气球、蹦极、巨人等特殊机制。
    for slot, zombie in enumerate(zombies[:MAX_ZOMBIES]):
        is_boss = int(_number(_field(zombie, "type", -1), -1)) == ZOMBIE_BOSS_TYPE
        row = int(_number(_field(zombie, "row", -1), -1))
        target_row = int(_number(_field(zombie, "target_row", -1), -1))
        target_col = int(_number(_field(zombie, "target_col", -1), -1))
        result["zombie_mask"][slot] = 1
        result["zombie_type"][slot] = _category(_field(zombie, "type", -1))
        result["zombie_phase"][slot] = _category(
            _field(zombie, "phase", _field(zombie, "state", -1)), STATE_CAPACITY - 1
        )
        result["zombie_height"][slot] = _category(
            _field(zombie, "height", -1), STATE_CAPACITY - 1
        )
        result["zombie_row"][slot] = row + 1 if not is_boss and 0 <= row < MAX_ROWS else 0
        result["zombie_target_row"][slot] = target_row + 1 if 0 <= target_row < MAX_ROWS else 0
        result["zombie_target_col"][slot] = target_col + 1 if 0 <= target_col < COLS else 0
        result["zombies"][slot] = (
            # PvZ 横向活动范围约 0～900；只缩放不截断，保留特殊越界位置。
            0.0 if is_boss else _number(_field(zombie, "x", 0)) / 900.0,
            0.0 if is_boss else _number(_field(zombie, "y", 0)) / 600.0,
            _number(_field(zombie, "vel_x", 0)) / 5.0,
            _number(_field(zombie, "altitude", 0)) / 3000.0,
            # 本体、头盔、盾牌、飞行耐久分开，避免混淆路障/铁桶、铁门、气球等机制。
            _log_scale(_field(zombie, "body_hp", _field(zombie, "hp", 0)), 20000.0),
            _log_scale(_field(zombie, "max_hp", 0), 20000.0),
            _log_scale(_field(zombie, "helm_hp", 0), 20000.0),
            _log_scale(_field(zombie, "shield_hp", 0), 20000.0),
            _log_scale(_field(zombie, "flying_hp", 0), 20000.0),
            # 行为阶段计时跨度较大；三个控制计时器保留减速、黄油和冰封剩余时间。
            _signed_log_scale(_field(zombie, "phase_counter", 0), 1000.0),
            _log_scale(_field(zombie, "chilled_counter", 0), 1000.0),
            _log_scale(_field(zombie, "buttered_counter", 0), 1000.0),
            _log_scale(_field(zombie, "ice_trap_counter", 0), 1000.0),
            float(bool(_field(zombie, "is_eating", False))),
            float(bool(_field(zombie, "is_mind_controlled", False))),
        )

    if boss_projectiles:
        projectiles = [p for p in _valid_boss_projectiles(raw_obs) if visible(p[1], p[2])]
        for slot, (kind, row, x, y) in enumerate(projectiles[:MAX_BOSS_PROJECTILES]):
            result["boss_projectile_type"][slot] = kind + 1
            result["boss_projectile_row"][slot] = row + 1
            result["boss_projectiles"][slot] = (x / 900.0, y / 600.0)
            result["boss_projectile_mask"][slot] = 1

    dropped = list(_field(raw_obs, "dropped_cards", ()) or ())
    dropped = [card for card in dropped if visible(
        int(np.clip(np.floor((_number(_field(card, "y", 0)) - 80.0) / 85.0), 0, MAX_ROWS - 1)),
        _number(_field(card, "x", 0)))]
    dropped.sort(key=lambda card: (_number(_field(card, "x", 0)), _number(_field(card, "y", 0))))
    for slot, card in enumerate(dropped[:MAX_DROPPED_CARDS]):
        result["dropped_card_type"][slot] = _category(_field(card, "type", -1))
        result["dropped_cards"][slot] = (
            1.0,
            float(np.clip(_number(_field(card, "x", 0)) / 900.0, -0.25, 1.5)),
            float(np.clip(_number(_field(card, "y", 0)) / 600.0, -0.25, 1.5)),
            1.0 if bool(_field(card, "is_collectable", False)) else 0.0,
        )

    mower_index = {"LOST": 1, "READY": 2, "ROLLING": 3}
    for mower in _field(raw_obs, "mowers", ()) or ():
        row = int(_number(_field(mower, "row", -1), -1))
        if 0 <= row < MAX_ROWS:
            result["mower_state"][row] = mower_index.get(
                str(_field(mower, "state", "LOST")).upper(), 1
            )

    for value in result.values():
        if np.issubdtype(value.dtype, np.floating):
            np.nan_to_num(value, copy=False, nan=0.0, posinf=4.0, neginf=-1.0)
            np.clip(value, -4.0, 4.0, out=value)
    return result


def observation_is_valid(value: object, *, boss_projectiles: bool = True) -> bool:
    spec = OBSERVATION_SPEC if boss_projectiles else LEGACY_OBSERVATION_SPEC
    if not isinstance(value, Mapping) or set(value) != set(spec):
        return False
    for name, (shape, dtype) in spec.items():
        array = value[name]
        if not isinstance(array, np.ndarray) or array.shape != shape or array.dtype != dtype:
            return False
        if np.issubdtype(dtype, np.floating) and not bool(np.isfinite(array).all()):
            return False
    return True


def make_observation_space(*, boss_projectiles: bool = True):
    spec = OBSERVATION_SPEC if boss_projectiles else LEGACY_OBSERVATION_SPEC
    try:
        from gymnasium import spaces

        fields = {
            "global": spaces.Box(-4.0, 4.0, (GLOBAL_FEATURES,), np.float32),
            "scene": spaces.Box(0, SCENE_CAPACITY, (1,), np.int64),
            "zombie_roster": spaces.MultiBinary(ZOMBIE_ROSTER_SIZE),
            "card_type": spaces.Box(0, TYPE_CAPACITY, (MAX_CARD_SLOTS,), np.int64),
            "cards": spaces.Box(-1.0, 4.0, (MAX_CARD_SLOTS, CARD_FEATURES), np.float32),
            "plant_type": spaces.Box(0, TYPE_CAPACITY, (PLANT_LAYERS, MAX_ROWS, COLS), np.int64),
            "plant_state": spaces.Box(0, STATE_CAPACITY, (PLANT_LAYERS, MAX_ROWS, COLS), np.int64),
            "plants": spaces.Box(-1.0, 4.0, (PLANT_LAYERS, MAX_ROWS, COLS, PLANT_FEATURES), np.float32),
            "grid_terrain": spaces.Box(0, TYPE_CAPACITY, (MAX_ROWS, COLS), np.int64),
            "grid": spaces.Box(-1.0, 4.0, (MAX_ROWS, COLS, GRID_FEATURES), np.float32),
            "zombie_grid": spaces.Box(0.0, 4.0, (MAX_ROWS, COLS, ZOMBIE_GRID_FEATURES), np.float32),
            "zombie_type": spaces.Box(0, TYPE_CAPACITY, (MAX_ZOMBIES,), np.int64),
            "zombie_phase": spaces.Box(0, STATE_CAPACITY, (MAX_ZOMBIES,), np.int64),
            "zombie_height": spaces.Box(0, STATE_CAPACITY, (MAX_ZOMBIES,), np.int64),
            "zombie_target_row": spaces.Box(0, MAX_ROWS, (MAX_ZOMBIES,), np.int64),
            "zombie_target_col": spaces.Box(0, COLS, (MAX_ZOMBIES,), np.int64),
            "zombie_row": spaces.Box(0, MAX_ROWS, (MAX_ZOMBIES,), np.int64),
            "zombies": spaces.Box(-4.0, 4.0, (MAX_ZOMBIES, ZOMBIE_FEATURES), np.float32),
            "zombie_mask": spaces.MultiBinary(MAX_ZOMBIES),
            "dropped_card_type": spaces.Box(0, TYPE_CAPACITY, (MAX_DROPPED_CARDS,), np.int64),
            "dropped_cards": spaces.Box(-1.0, 4.0, (MAX_DROPPED_CARDS, DROPPED_CARD_FEATURES), np.float32),
            "mower_state": spaces.Box(0, MOWER_STATES - 1, (MAX_ROWS,), np.int64),
            "boss_projectile_type": spaces.Box(0, 2, (MAX_BOSS_PROJECTILES,), np.int64),
            "boss_projectile_row": spaces.Box(0, MAX_ROWS, (MAX_BOSS_PROJECTILES,), np.int64),
            "boss_projectiles": spaces.Box(-4.0, 4.0, (MAX_BOSS_PROJECTILES, BOSS_PROJECTILE_FEATURES), np.float32),
            "boss_projectile_mask": spaces.MultiBinary(MAX_BOSS_PROJECTILES),
        }
        return spaces.Dict({name: field for name, field in fields.items() if name in spec})
    except ImportError:  # pragma: no cover
        class DictSpace:
            shape = None
            dtype = None
            spaces = spec

            @staticmethod
            def contains(value: object) -> bool:
                return observation_is_valid(value, boss_projectiles=boss_projectiles)

        return DictSpace()


__all__ = [
    "OBSERVATION_SCHEMA",
    "OBSERVATION_SPEC",
    "LEGACY_OBSERVATION_SCHEMA",
    "LEGACY_OBSERVATION_SPEC",
    "MAX_BOSS_PROJECTILES",
    "BOSS_PROJECTILE_FIELDS",
    "BOSS_PROJECTILE_FEATURES",
    "MAX_ZOMBIES",
    "ZOMBIE_CONTINUOUS_FIELDS",
    "ZOMBIE_GRID_FEATURES",
    "zombie_truncation_info",
    "boss_projectile_truncation_info",
    "encode_observation",
    "empty_observation",
    "make_observation_space",
    "observation_is_valid",
]
