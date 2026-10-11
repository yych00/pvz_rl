"""Enhanced Tactical Generic Policy for all PVZ Adventure Levels.

Provides a robust, role-based, multi-stage tactical policy for any level
without a specialized script. Replaces naive first-legal placement with
intelligent positioning:
- Economy in back (col 0~1)
- Shooters in mid-back (col 1~3), prioritizing threatened lanes
- Wall-nuts/Tall-nuts in front (col 4~6)
- Instant explosives strictly reserved for active zombie threats (no empty detonations)
- Grave Busters prioritize graves
- Environment supports (Lily Pad, Flower Pot) prioritize required terrain
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

from .conveyor import cards_by_type, lane_health, legal_choice, row_count, zombies_by_row
from .level_1_2 import ACTION_SPACE_SIZE, COLS, MAX_CARD_SLOTS, MAX_ROWS, Choice

# Plant Type IDs
PEASHOOTER = 0
SUNFLOWER = 1
CHERRY_BOMB = 2
WALL_NUT = 3
POTATO_MINE = 4
SNOW_PEA = 5
CHOMPER = 6
REPEATER = 7
PUFF_SHROOM = 8
SUN_SHROOM = 9
FUME_SHROOM = 10
GRAVE_BUSTER = 11
HYPNO_SHROOM = 12
SCAREDY_SHROOM = 13
ICE_SHROOM = 14
DOOM_SHROOM = 15
LILY_PAD = 16
SQUASH = 17
THREEPEATER = 18
TANGLE_KELP = 19
JALAPENO = 20
SPIKEWEED = 21
TORCHWOOD = 22
TALL_NUT = 23
SEA_SHROOM = 24
PLANTERN = 25
CACTUS = 26
BLOVER = 27
SPLIT_PEA = 28
STARFRUIT = 29
PUMPKIN = 30
MAGNET_SHROOM = 31
CABBAGE_PULT = 32
FLOWER_POT = 33
KERNEL_PULT = 34
COFFEE_BEAN = 35
GARLIC = 36
UMBRELLA_LEAF = 37
MARIGOLD = 38
MELON_PULT = 39
GATLING_PEA = 40
TWIN_SUNFLOWER = 41
GLOOM_SHROOM = 42
CATTAIL = 43
WINTER_MELON = 44
GOLD_MAGNET = 45
SPIKEROCK = 46
COB_CANNON = 47

# Role Classifications
ECONOMY_PLANTS = (SUNFLOWER, SUN_SHROOM, TWIN_SUNFLOWER)
SHOOTERS = (
    REPEATER,
    SNOW_PEA,
    PEASHOOTER,
    MELON_PULT,
    WINTER_MELON,
    GATLING_PEA,
    KERNEL_PULT,
    CABBAGE_PULT,
    FUME_SHROOM,
    THREEPEATER,
    CACTUS,
    SCAREDY_SHROOM,
    STARFRUIT,
    CATTAIL,
)
WALL_PLANTS = (WALL_NUT, TALL_NUT, PUMPKIN)
FODDER_PLANTS = (PUFF_SHROOM, SEA_SHROOM)
INSTANT_EXPLOSIVE_PLANTS = {
    CHERRY_BOMB,
    POTATO_MINE,
    ICE_SHROOM,
    DOOM_SHROOM,
    SQUASH,
    TANGLE_KELP,
    JALAPENO,
}

DANGEROUS_ZOMBIES = frozenset((4, 7, 12, 14, 15, 19, 20))  # Buckethead, Football, etc.

# High-winrate Adventure Deck Priority (King Deck)
ENHANCED_GENERIC_DECK_PRIORITY: Tuple[int, ...] = (
    # 1. Economy Core
    SUNFLOWER,
    SUN_SHROOM,
    TWIN_SUNFLOWER,
    # 2. Defense and Environmental Support
    WALL_NUT,
    TALL_NUT,
    LILY_PAD,
    FLOWER_POT,
    GRAVE_BUSTER,
    # 3. Core Firepower
    REPEATER,
    SNOW_PEA,
    PEASHOOTER,
    MELON_PULT,
    WINTER_MELON,
    KERNEL_PULT,
    CABBAGE_PULT,
    FUME_SHROOM,
    CACTUS,
    # 4. Emergency & Area Explosives
    SQUASH,
    CHERRY_BOMB,
    JALAPENO,
    POTATO_MINE,
    ICE_SHROOM,
    DOOM_SHROOM,
    TANGLE_KELP,
    # 5. Strategic Support & Upgrades
    TORCHWOOD,
    MAGNET_SHROOM,
    PUMPKIN,
    SPIKEWEED,
    SPIKEROCK,
    BLOVER,
    PUFF_SHROOM,
    GATLING_PEA,
    CATTAIL,
    GLOOM_SHROOM,
    # 6. Remainder fallback
    *tuple(
        i
        for i in range(64)
        if i
        not in (
            SUNFLOWER,
            SUN_SHROOM,
            TWIN_SUNFLOWER,
            WALL_NUT,
            TALL_NUT,
            LILY_PAD,
            FLOWER_POT,
            GRAVE_BUSTER,
            REPEATER,
            SNOW_PEA,
            PEASHOOTER,
            MELON_PULT,
            WINTER_MELON,
            KERNEL_PULT,
            CABBAGE_PULT,
            FUME_SHROOM,
            CACTUS,
            SQUASH,
            CHERRY_BOMB,
            JALAPENO,
            POTATO_MINE,
            ICE_SHROOM,
            DOOM_SHROOM,
            TANGLE_KELP,
            TORCHWOOD,
            MAGNET_SHROOM,
            PUMPKIN,
            SPIKEWEED,
            SPIKEROCK,
            BLOVER,
            PUFF_SHROOM,
            GATLING_PEA,
            CATTAIL,
            GLOOM_SHROOM,
        )
    ),
)


def _zombie_col(x: float) -> int:
    return max(0, min(8, int((x - 40) // 80)))


def choose_enhanced_generic_plant(mask: Sequence[bool], obs: dict) -> Optional[Choice]:
    """Smart role-based generic tactical policy for all adventure levels."""
    if len(mask) != ACTION_SPACE_SIZE:
        return None

    rows = row_count(obs)
    cards = cards_by_type(obs)
    lanes = zombies_by_row(obs, rows)

    # 1. 结构化场上植物与格子物件
    plants_by_cell: Dict[Tuple[int, int], List[dict]] = {}
    for plant in obs.get("plants", []):
        if isinstance(plant, dict):
            r = int(plant.get("row", -1))
            c = int(plant.get("col", -1))
            if 0 <= r < rows and 0 <= c < 9:
                plants_by_cell.setdefault((r, c), []).append(plant)

    graves_by_row: Dict[int, List[int]] = {}
    vases_by_cell: Set[Tuple[int, int]] = set()
    for item in obs.get("grid_items", []):
        if not isinstance(item, dict):
            continue
        itype = str(item.get("type", ""))
        r = int(item.get("row", -1))
        c = int(item.get("col", -1))
        if 0 <= r < rows and 0 <= c < 9:
            if itype == "GRAVE":
                graves_by_row.setdefault(r, []).append(c)
            elif itype == "VASE":
                vases_by_cell.add((r, c))

    # 统计每行现有单位数量
    shooters_in_row = [
        sum(int(p.get("type", -1)) in SHOOTERS for c in range(9) for p in plants_by_cell.get((r, c), []))
        for r in range(rows)
    ]
    walls_in_row = [
        sum(int(p.get("type", -1)) in WALL_PLANTS for c in range(9) for p in plants_by_cell.get((r, c), []))
        for r in range(rows)
    ]
    eco_count = sum(
        sum(int(p.get("type", -1)) in ECONOMY_PLANTS for c in range(9) for p in plants_by_cell.get((r, c), []))
        for r in range(rows)
    )

    # 威胁排序：按最靠前僵尸的 x 升序排序
    threatened = sorted(
        (r for r in range(rows) if lanes[r]),
        key=lambda r: (
            float(lanes[r][0].get("x", 9999) or 9999),
            -len(lanes[r]),
            -lane_health(lanes[r]),
        ),
    )

    # -------------------------------------------------------------------------
    # 策略 1: 墓碑吞噬 (Grave Buster)
    # -------------------------------------------------------------------------
    if GRAVE_BUSTER in cards and graves_by_row:
        # 优先吞噬靠右或有僵尸行的墓碑
        for r in threatened + [r for r in range(rows) if r not in threatened]:
            if r in graves_by_row:
                target_cols = sorted(graves_by_row[r], reverse=True)
                choice = legal_choice(mask, cards, GRAVE_BUSTER, (r,), target_cols, "generic-eat-grave")
                if choice:
                    return choice

    # -------------------------------------------------------------------------
    # 策略 2: 紧急救场（严禁无僵尸空放炸弹！）
    # -------------------------------------------------------------------------
    for r in threatened:
        nearest_x = float(lanes[r][0].get("x", 9999) or 9999)
        zcount = len(lanes[r])
        health = lane_health(lanes[r])
        has_armored = any(int(z.get("type", -1)) in DANGEROUS_ZOMBIES for z in lanes[r])

        # 水路潜水/逼近：水草单杀
        if r in (2, 3) and rows == 6 and nearest_x < 650:
            zcol = _zombie_col(nearest_x)
            choice = legal_choice(mask, cards, TANGLE_KELP, (r,), (zcol, zcol + 1), "generic-tangle-kelp")
            if choice:
                return choice

        # 窝瓜近身单压 (x < 420)
        if nearest_x < 420:
            zcol = _zombie_col(nearest_x)
            choice = legal_choice(mask, cards, SQUASH, (r,), (zcol, zcol + 1), "generic-squash-urgent")
            if choice:
                return choice

        # 土豆雷埋雷：在僵尸距离尚可 (450 < x < 680) 且前方有准备时间时埋设
        if 450 < nearest_x < 680 and shooters_in_row[r] == 0:
            choice = legal_choice(mask, cards, POTATO_MINE, (r,), (3, 2, 4), "generic-potato-mine")
            if choice:
                return choice

        # 樱桃炸弹：逼近危险线 (x < 360) 或装甲/扎堆尸群
        if nearest_x < 360 or has_armored or (zcount >= 2 and health >= 550):
            zcol = _zombie_col(nearest_x)
            blast_rows = [row for row in (r, r - 1, r + 1) if 0 <= row < rows]
            blast_cols = [col for col in (zcol - 1, zcol, zcol + 1) if 0 <= col < 9]
            choice = legal_choice(mask, cards, CHERRY_BOMB, blast_rows, blast_cols, "generic-cherry-emergency")
            if choice:
                return choice

        # 火爆辣椒：单路尸潮全灭 (x < 320 或 zcount >= 3)
        if (nearest_x < 320 or zcount >= 3) and health >= 600:
            choice = legal_choice(mask, cards, JALAPENO, (r,), (0, 1, 2, 3, 4), "generic-jalapeno-line")
            if choice:
                return choice

        # 寒冰菇控场
        if nearest_x < 260 or (len(threatened) >= 3 and health >= 1000):
            choice = legal_choice(mask, cards, ICE_SHROOM, range(rows), range(9), "generic-ice-freeze")
            if choice:
                return choice

        # 毁灭菇大招清屏
        if nearest_x < 220 or (zcount >= 4 and health >= 1400):
            choice = legal_choice(mask, cards, DOOM_SHROOM, (2, 1, 3), (3, 4, 5), "generic-doom-nuke")
            if choice:
                return choice

    # -------------------------------------------------------------------------
    # 策略 3: 环境载具铺设 (睡莲 / 花盆)
    # -------------------------------------------------------------------------
    if rows == 6 and LILY_PAD in cards:
        # 泳池 6 行：在水路 (row 2, 3) 铺设睡莲
        for r in (2, 3):
            for c in (1, 2, 3, 4):
                if (r, c) not in plants_by_cell and (r, c) not in vases_by_cell:
                    choice = legal_choice(mask, cards, LILY_PAD, (r,), (c,), "generic-lily-pad")
                    if choice:
                        return choice

    if FLOWER_POT in cards:
        # 屋顶关卡：铺设花盆
        for r in (2, 1, 3, 0, 4):
            if r >= rows:
                continue
            for c in (1, 2, 3, 4, 5):
                if (r, c) not in plants_by_cell and (r, c) not in vases_by_cell:
                    choice = legal_choice(mask, cards, FLOWER_POT, (r,), (c,), "generic-flower-pot")
                    if choice:
                        return choice

    # -------------------------------------------------------------------------
    # 策略 4: 濒危行火力覆盖（保护已有僵尸逼近但无输出的行）
    # -------------------------------------------------------------------------
    lacking_firepower = [r for r in threatened if shooters_in_row[r] == 0]
    if lacking_firepower:
        for plant_type in SHOOTERS:
            for r in lacking_firepower:
                for c in (1, 2, 3, 0):
                    if (r, c) not in vases_by_cell:
                        choice = legal_choice(mask, cards, plant_type, (r,), (c,), "generic-cover-active-lane")
                        if choice:
                            return choice

    # -------------------------------------------------------------------------
    # 策略 5: 经济基础优先构建（向日葵 / 阳光菇永远种在后排安全区 col 0~1）
    # -------------------------------------------------------------------------
    if eco_count < 8:
        # 优先从中间行向上下扩展后排
        center_rows = sorted(range(rows), key=lambda r: (abs(r - (rows - 1) / 2), r))
        for plant_type in ECONOMY_PLANTS:
            for c in (0, 1):
                for r in center_rows:
                    if (r, c) not in vases_by_cell:
                        choice = legal_choice(mask, cards, plant_type, (r,), (c,), "generic-economy-backline")
                        if choice:
                            return choice

    # -------------------------------------------------------------------------
    # 策略 6: 前线坚果拦截 (col 4~6)
    # -------------------------------------------------------------------------
    for r in threatened:
        if float(lanes[r][0].get("x", 9999) or 9999) < 650 and walls_in_row[r] == 0:
            for plant_type in WALL_PLANTS:
                for c in (5, 4, 6, 3):
                    if (r, c) not in vases_by_cell:
                        choice = legal_choice(mask, cards, plant_type, (r,), (c,), "generic-wall-block")
                        if choice:
                            return choice

    # -------------------------------------------------------------------------
    # 策略 7: 稳步建立全线基础防线 (各路火力均衡)
    # -------------------------------------------------------------------------
    for plant_type in SHOOTERS:
        for r in range(rows):
            if shooters_in_row[r] < 2:
                for c in (1, 2, 3, 0):
                    if (r, c) not in vases_by_cell:
                        choice = legal_choice(mask, cards, plant_type, (r,), (c,), "generic-build-line")
                        if choice:
                            return choice

    # -------------------------------------------------------------------------
    # 策略 8: 垫材布置与传送带防卡死泄洪 (小喷菇、海蘑菇)
    # -------------------------------------------------------------------------
    total_cards = sum(len(slots) for slots in cards.values())
    if total_cards >= 4:
        for fodder in FODDER_PLANTS:
            for r in range(rows):
                for c in (4, 5, 6, 7, 3):
                    if (r, c) not in vases_by_cell:
                        choice = legal_choice(mask, cards, fodder, (r,), (c,), "generic-fodder-flush")
                        if choice:
                            return choice

    # -------------------------------------------------------------------------
    # 策略 9: 保底兜底（仅限非爆炸植物，按安全区遍历，确保不卡死）
    # -------------------------------------------------------------------------
    row_order = [r for r in (2, 1, 3, 0, 4, 5) if r < rows]
    col_order = (1, 2, 0, 3, 4, 5, 6, 7, 8)
    for ptype, slots in sorted(cards.items()):
        if ptype in INSTANT_EXPLOSIVE_PLANTS:
            continue
        for slot in slots:
            for row in row_order:
                for col in col_order:
                    if (row, col) in vases_by_cell:
                        continue
                    idx = 1 + ((slot * MAX_ROWS + row) * COLS + col)
                    if 0 <= idx < len(mask) and bool(mask[idx]):
                        return idx, slot, row, col, ptype, "generic-safe-fallback"

    return None
