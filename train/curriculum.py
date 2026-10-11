"""Adventure curriculum and mixed level pool management."""

from __future__ import annotations

from typing import Sequence
import numpy as np

SCENE_NAMES = {
    1: "白天草地 (1-x)",
    2: "黑夜草地 (2-x)",
    3: "白天泳池 (3-x)",
    4: "浓雾泳池 (4-x)",
    5: "屋顶花园 (5-x)",
}

REGULAR_SUBS = (1, 2, 3, 4, 6, 7, 8, 9)
FUN_SUB = 5
CONVEYOR_SUB = 10


def format_level_label(level_id: int) -> str:
    """将内部 level_id (1..50) 格式化为大关-小关 (如 1-4, 2-1)"""
    val = int(level_id)
    if 1 <= val <= 50:
        area = (val - 1) // 10 + 1
        sub = (val - 1) % 10 + 1
        return f"{area}-{sub}"
    return str(val)


def generate_mixed_level_pool(
    scenes: Sequence[int],
    playthroughs: Sequence[int],
    include_fun: bool = False,
    include_conveyor: bool = False,
) -> tuple[tuple[int, int], ...]:
    """生成混合关卡池列表，每项为 (level_id, playthrough)

    参数:
        scenes: 选中的场景列表 (1..5)
        playthroughs: 选中的周目列表 (1..2)
        include_fun: 是否包含 x-5 趣味关卡 (坚果保龄球、打僵尸、小僵尸、砸罐子等)
        include_conveyor: 是否包含 x-10 传送带关卡 (草地传送带、夜晚传送带等)
    """
    valid_scenes = sorted(set(int(s) for s in scenes if 1 <= int(s) <= 5))
    valid_playthroughs = sorted(set(int(p) for p in playthroughs if int(p) in (1, 2)))
    if not valid_scenes or not valid_playthroughs:
        return ()

    subs = list(REGULAR_SUBS)
    if include_fun:
        subs.append(FUN_SUB)
    if include_conveyor:
        subs.append(CONVEYOR_SUB)
    subs.sort()

    pool: list[tuple[int, int]] = []
    for playthrough in valid_playthroughs:
        for scene in valid_scenes:
            for sub in subs:
                level_id = (scene - 1) * 10 + sub
                pool.append((level_id, playthrough))
    return tuple(pool)


def format_level_pool_summary(pool: Sequence[tuple[int, int]]) -> str:
    """生成关卡池的人类可读摘要"""
    if not pool:
        return "空 (未选中任何关卡)"

    p_set = sorted(set(p for _, p in pool))
    p_text = "、".join(f"{p}周目" for p in p_set)

    # 统计涉及的大关
    scenes = sorted(set((lvl - 1) // 10 + 1 for lvl, _ in pool))
    scene_labels = [f"{s}大关" for s in scenes]

    return f"共 {len(pool)} 关 [{', '.join(scene_labels)} | {p_text}]"


def sample_level_from_pool(
    pool: Sequence[tuple[int, int]],
    rng: np.random.Generator | None = None,
) -> tuple[int, int]:
    """从关卡池中均匀随机抽选一个关卡 (level_id, playthrough)"""
    if not pool:
        raise ValueError("level pool is empty")
    if rng is None:
        idx = int(np.random.randint(0, len(pool)))
    else:
        idx = int(rng.integers(0, len(pool)))
    return pool[idx]


def episode_level_metadata(info, env_config, env=None) -> dict:
    """Freeze reset-time identity before later resets change the environment."""
    return {
        "level_id": info.get("level_id", getattr(env, "level_id", env_config.level_id)),
        "playthrough": info.get("playthrough", getattr(env, "playthrough", env_config.playthrough)),
        "episode_seed": info.get("episode_seed"),
        "seed_selection": dict(info.get("seed_selection") or {}),
    }
