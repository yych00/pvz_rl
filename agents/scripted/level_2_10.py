# -*- coding: utf-8 -*-
"""Adventure 2-10 Night conveyor policy.

Night conveyor belt provides mushrooms (Puff-shroom, Fume-shroom, Scaredy-shroom,
Hypno-shroom, Ice-shroom, Doom-shroom), Grave Buster, and Wall-nut.
No sun cost; fast packet consumption prevents conveyor belt stalls.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .conveyor import cards_by_type, lane_health, legal_choice, row_count, zombies_by_row
from .level_1_2 import ACTION_SPACE_SIZE, Choice

# Plant type IDs
WALL_NUT = 3
PUFF_SHROOM = 8
SUN_SHROOM = 9
FUME_SHROOM = 10
GRAVE_BUSTER = 11
HYPNO_SHROOM = 12
SCAREDY_SHROOM = 13
ICE_SHROOM = 14
DOOM_SHROOM = 15

DANGEROUS_ZOMBIES = frozenset((4, 7, 12, 15))  # Buckethead, Football, Newspaper enraged, Gargantuar


def _zombie_col(x: float) -> int:
    return max(0, min(8, int((x - 40) // 80)))


def choose_level_2_10_plant(mask: Sequence[bool], obs: dict) -> Optional[Choice]:
    if len(mask) != ACTION_SPACE_SIZE:
        return None

    rows = row_count(obs)
    cards = cards_by_type(obs)
    lanes = zombies_by_row(obs, rows)

    plants_by_cell: dict[tuple[int, int], list[dict]] = {}
    for plant in obs.get("plants", []):
        if isinstance(plant, dict):
            r = int(plant.get("row", -1))
            c = int(plant.get("col", -1))
            if 0 <= r < rows and 0 <= c < 9:
                plants_by_cell.setdefault((r, c), []).append(plant)

    center = sorted(range(rows), key=lambda r: (abs(r - (rows - 1) / 2), r))
    threatened = sorted(
        (r for r in range(rows) if lanes[r]),
        key=lambda r: (
            float(lanes[r][0].get("x", 9999) or 9999),
            -len(lanes[r]),
            -lane_health(lanes[r]),
        ),
    )

    # 1. 优先使用墓碑吞噬者清除场上的墓碑
    graves: list[tuple[int, int]] = []
    for item in obs.get("grid_items", []):
        if isinstance(item, dict) and item.get("type") == "GRAVE":
            gr = int(item.get("row", -1))
            gc = int(item.get("col", -1))
            if 0 <= gr < rows and 0 <= gc < 9:
                graves.append((gr, gc))
    if graves and GRAVE_BUSTER in cards:
        # 优先清除靠左（距离防线近）的墓碑
        graves.sort(key=lambda pos: (pos[1], pos[0]))
        for gr, gc in graves:
            choice = legal_choice(
                mask, cards, GRAVE_BUSTER, (gr,), (gc,), "2-10-bust-grave"
            )
            if choice:
                return choice

    # 2. 危机控场：大波僵尸或严重威胁时使用毁灭菇 / 寒冰菇
    is_huge = bool(obs.get("global", {}).get("is_huge_wave", False))
    total_zombies = sum(len(lane) for lane in lanes)
    critical_lane = any(
        lanes[r] and float(lanes[r][0].get("x", 9999) or 9999) < 220
        for r in range(rows)
    )

    if (is_huge or total_zombies >= 6 or critical_lane) and ICE_SHROOM in cards:
        # 寒冰菇全场减速冻结，放前排或空位
        for r in center:
            for c in (4, 3, 5, 2, 6):
                choice = legal_choice(
                    mask, cards, ICE_SHROOM, (r,), (c,), "2-10-freeze-all"
                )
                if choice:
                    return choice

    if (is_huge or total_zombies >= 8 or critical_lane) and DOOM_SHROOM in cards:
        # 毁灭菇大范围清场，放靠近僵尸群的列 (4~6)
        for r in threatened or center:
            for c in (4, 5, 3, 6):
                choice = legal_choice(
                    mask, cards, DOOM_SHROOM, (r,), (c,), "2-10-doom-clear"
                )
                if choice:
                    return choice

    # 3. 魅惑菇：遇到强力威胁（橄榄球/铁桶）在前方放置
    if HYPNO_SHROOM in cards:
        for r in threatened:
            for z in lanes[r]:
                zx = float(z.get("x", 9999) or 9999)
                ztype = int(z.get("type", -1))
                if ztype in DANGEROUS_ZOMBIES and 160 <= zx <= 560:
                    zcol = _zombie_col(zx)
                    target_cols = [c for c in (zcol - 1, zcol, zcol - 2) if 0 <= c < 9]
                    choice = legal_choice(
                        mask, cards, HYPNO_SHROOM, (r,), target_cols, "2-10-hypno-threat"
                    )
                    if choice:
                        return choice

    # 4. 坚果墙：在第 5、4 列阻挡僵尸
    if WALL_NUT in cards:
        for r in threatened:
            has_wall = any(
                int(p.get("type", -1)) == WALL_NUT
                for (pr, pc), plist in plants_by_cell.items()
                if pr == r
                for p in plist
            )
            if not has_wall:
                choice = legal_choice(
                    mask, cards, WALL_NUT, (r,), (5, 4, 6), "2-10-block-lane"
                )
                if choice:
                    return choice

    # 5. 胆小菇：必须放在最安全的后方（第 0, 1 列）
    if SCAREDY_SHROOM in cards:
        # 优先填补未被僵尸贴近的后方行
        safe_rows = [
            r for r in center
            if not lanes[r] or float(lanes[r][0].get("x", 9999) or 9999) > 280
        ]
        choice = legal_choice(
            mask, cards, SCAREDY_SHROOM, safe_rows or center, (0, 1), "2-10-sniper-scaredy"
        )
        if choice:
            return choice

    # 6. 大喷菇：穿透输出，布置在第 1~3 列
    if FUME_SHROOM in cards:
        choice = legal_choice(
            mask, cards, FUME_SHROOM, threatened or center, (1, 2, 3), "2-10-fume-line"
        )
        if choice:
            return choice

    # 7. 小喷菇：0 阳光免费近战输出，快速倾泻在 2~6 列，防止传送带堵塞
    if PUFF_SHROOM in cards:
        choice = legal_choice(
            mask, cards, PUFF_SHROOM, threatened or center, (3, 4, 2, 5, 6, 1), "2-10-puff-spam"
        )
        if choice:
            return choice

    # 8. 阳光菇（如果有）：放在后排
    if SUN_SHROOM in cards:
        choice = legal_choice(
            mask, cards, SUN_SHROOM, center, (0, 1), "2-10-sun-shroom"
        )
        if choice:
            return choice

    # 9. 传送带满溢保护：只要手上有卡且有合法格子，积极消费最左侧卡片，释放传送带容量
    for ptype in cards:
        choice = legal_choice(
            mask, cards, ptype, center, range(9), "2-10-conveyor-drain"
        )
        if choice:
            return choice

    return None
