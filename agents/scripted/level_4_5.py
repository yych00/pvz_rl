"""Adventure 4-5 Vasebreaker (砸罐子) scripted policy.

4-5 is a puzzle mini-game where the lawn is covered with mystery vases.
Breaking vases releases plants or zombies.
This policy intelligently distributes rescued plants and strictly holds
high-value emergency explosives (Cherry Bomb) until dangerous zombies appear.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .conveyor import cards_by_type, lane_health, legal_choice, row_count, zombies_by_row
from .level_1_2 import ACTION_SPACE_SIZE, Choice

# Plant Type IDs
PEASHOOTER = 0
CHERRY_BOMB = 2
WALL_NUT = 3
POTATO_MINE = 4
SNOW_PEA = 5
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
PLANTERN = 25

SHOOTERS = (REPEATER, SNOW_PEA, PEASHOOTER, FUME_SHROOM)
DANGEROUS_ZOMBIES = frozenset((4, 7, 12, 14, 15))  # Buckethead, Football, etc.
INTERACT_BASE = 703


def _zombie_col(x: float) -> int:
    return max(0, min(8, int((x - 40) // 80)))


def choose_level_4_5_plant(mask: Sequence[bool], obs: dict) -> Optional[Choice]:
    """Tactical policy for 4-5 Vasebreaker."""
    if len(mask) != ACTION_SPACE_SIZE:
        return None

    rows = row_count(obs)
    cards = cards_by_type(obs)
    lanes = zombies_by_row(obs, rows)

    # 提取场上现有植物与罐子位置
    plants_by_cell: dict[tuple[int, int], list[dict]] = {}
    for plant in obs.get("plants", []):
        if isinstance(plant, dict):
            r = int(plant.get("row", -1))
            c = int(plant.get("col", -1))
            if 0 <= r < rows and 0 <= c < 9:
                plants_by_cell.setdefault((r, c), []).append(plant)

    vases_by_cell: set[tuple[int, int]] = set()
    for item in obs.get("grid_items", []):
        if isinstance(item, dict) and item.get("type") == "VASE":
            r = int(item.get("row", -1))
            c = int(item.get("col", -1))
            if 0 <= r < rows and 0 <= c < 9:
                vases_by_cell.add((r, c))

    # 统计每行活跃僵尸与危险度
    threatened = sorted(
        (r for r in range(rows) if lanes[r]),
        key=lambda r: (
            float(lanes[r][0].get("x", 9999) or 9999),
            -len(lanes[r]),
            -lane_health(lanes[r]),
        ),
    )

    # -------------------------------------------------------------------------
    # 1. 紧急灭火：樱桃炸弹 / 窝瓜
    # 核心铁律：绝对禁止无僵尸时裸放樱桃炸弹！
    # 只有当：有僵尸逼近底线 (x < 380) 或该行出现铁桶/多只僵尸扎堆时才允许投掷！
    # -------------------------------------------------------------------------
    for r in threatened:
        nearest_x = float(lanes[r][0].get("x", 9999) or 9999)
        zcount = len(lanes[r])
        health = lane_health(lanes[r])
        has_armored = any(int(z.get("type", -1)) in DANGEROUS_ZOMBIES for z in lanes[r])

        # 窝瓜定点下压
        if nearest_x < 450:
            zcol = _zombie_col(nearest_x)
            choice = legal_choice(mask, cards, SQUASH, (r,), (zcol, zcol + 1), "4-5-squash-target")
            if choice:
                return choice

        # 樱桃炸弹：逼近危险线 (x < 380)，或者出现铁桶/群聚尸潮
        if nearest_x < 380 or has_armored or (zcount >= 2 and health >= 500):
            zcol = _zombie_col(nearest_x)
            blast_rows = [row for row in (r, r - 1, r + 1) if 0 <= row < rows]
            blast_cols = [col for col in (zcol - 1, zcol, zcol + 1) if 0 <= col < 9]
            choice = legal_choice(
                mask, cards, CHERRY_BOMB, blast_rows, blast_cols, "4-5-cherry-emergency"
            )
            if choice:
                return choice

    # -------------------------------------------------------------------------
    # 2. 路灯花 (Plantern)：照亮罐子，优先放在中场已经清空的格子上
    # -------------------------------------------------------------------------
    has_plantern = any(
        any(int(p.get("type", -1)) == PLANTERN for p in plants_by_cell.get((r, c), []))
        for r in range(rows)
        for c in range(9)
    )
    if not has_plantern:
        for r in (2, 1, 3, 0, 4):
            if r >= rows:
                continue
            for c in (3, 4, 2, 5):
                if (r, c) not in vases_by_cell and (r, c) not in plants_by_cell:
                    choice = legal_choice(mask, cards, PLANTERN, (r,), (c,), "4-5-plantern-reveal")
                    if choice:
                        return choice

    # -------------------------------------------------------------------------
    # 3. 建立射手防线：将开出的射手布设在后排空地 (col 0~2)
    # 优先部署在已有僵尸活动且缺乏输出的行
    # -------------------------------------------------------------------------
    for shooter in SHOOTERS:
        # 先给有僵尸的行配齐射手
        for r in threatened:
            has_shooter = any(
                int(p.get("type", -1)) in SHOOTERS
                for c in range(9)
                for p in plants_by_cell.get((r, c), [])
            )
            if not has_shooter:
                for c in (0, 1, 2):
                    if (r, c) not in vases_by_cell and (r, c) not in plants_by_cell:
                        choice = legal_choice(mask, cards, shooter, (r,), (c,), "4-5-defend-lane-shooter")
                        if choice:
                            return choice

        # 再在后排空地预先筑造防线
        for r in range(rows):
            for c in (0, 1, 2):
                if (r, c) not in vases_by_cell and (r, c) not in plants_by_cell:
                    choice = legal_choice(mask, cards, shooter, (r,), (c,), "4-5-place-backline-shooter")
                    if choice:
                        return choice

    # -------------------------------------------------------------------------
    # 4. 坚果阻挡：放置在有僵尸逼近的路线上 (col 2~5)
    # -------------------------------------------------------------------------
    for r in threatened:
        if float(lanes[r][0].get("x", 9999) or 9999) < 650:
            for c in (3, 4, 2, 5):
                if (r, c) not in vases_by_cell and (r, c) not in plants_by_cell:
                    choice = legal_choice(mask, cards, WALL_NUT, (r,), (c,), "4-5-wall-nut-block")
                    if choice:
                        return choice

    # -------------------------------------------------------------------------
    # 5. 胆小菇与小喷菇：补刀与垫材
    # 胆小菇只能放在第 0 列；小喷菇可以放在僵尸前方阻挡
    # -------------------------------------------------------------------------
    for r in range(rows):
        if (r, 0) not in vases_by_cell and (r, 0) not in plants_by_cell:
            choice = legal_choice(mask, cards, SCAREDY_SHROOM, (r,), (0,), "4-5-scaredy-shroom")
            if choice:
                return choice

    for r in threatened:
        nearest_x = float(lanes[r][0].get("x", 9999) or 9999)
        zcol = _zombie_col(nearest_x)
        for c in (zcol, zcol - 1, zcol + 1):
            if 0 <= c < 9 and (r, c) not in vases_by_cell and (r, c) not in plants_by_cell:
                choice = legal_choice(mask, cards, PUFF_SHROOM, (r,), (c,), "4-5-puff-shroom-meatshield")
                if choice:
                    return choice

    # -------------------------------------------------------------------------
    # 6. 传送带泄洪（仅限无害廉价植物）：如果卡槽快满了，把小喷菇种在任何空地
    # 绝对不把樱桃炸弹当作泄洪卡！
    # -------------------------------------------------------------------------
    total_cards = sum(len(slots) for slots in cards.values())
    if total_cards >= 5:
        for r in range(rows):
            for c in range(9):
                if (r, c) not in vases_by_cell and (r, c) not in plants_by_cell:
                    choice = legal_choice(mask, cards, PUFF_SHROOM, (r,), (c,), "4-5-flush-puff")
                    if choice:
                        return choice

    # -------------------------------------------------------------------------
    # 7. 动作砸罐子 (Vasebreaker InteractCell: 703 + r * 9 + c)
    #
    # 战术准则：
    # 1. 控场节流：场上僵尸过多 (>= 2) 或有僵尸逼近底线 (x < 480) 时暂停开罐，
    #    让已部署的植物先击杀僵尸，避免尸潮叠加崩溃。
    # 2. 手牌未消化时不急于开罐：手里若有防御/攻击植物未种，优先在前面步骤种下。
    # 3. 选罐优先级：
    #    a. 从最右侧向左开 (Col 8 -> Col 0)：为开出的僵尸争取最长行进与射击时间；
    #    b. 优先开已有射手防线的行：开出普通/路障僵尸可被单解；
    #    c. 避开当前已有僵尸正在行进的行：防止该行多怪扎堆。
    # -------------------------------------------------------------------------

    # 安全闸门：场上若有活跃僵尸逼近半场，或存活僵尸总数 >= 2，绝不开新罐
    any_close_threat = any(
        float(lanes[r][0].get("x", 9999) or 9999) < 480
        for r in range(rows)
        if lanes[r]
    )
    total_active_zombies = sum(len(lanes[r]) for r in range(rows))
    if any_close_threat or total_active_zombies >= 2:
        return None

    # 收集当前可合法开罐的候选目标
    candidate_vases: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(9):
            if (r, c) in vases_by_cell:
                act_idx = INTERACT_BASE + r * 9 + c
                if act_idx < len(mask) and mask[act_idx]:
                    candidate_vases.append((r, c))

    if not candidate_vases:
        return None

    # 计算各行是否有射手
    lane_has_shooter = {
        r: any(
            int(p.get("type", -1)) in SHOOTERS
            for c in range(9)
            for p in plants_by_cell.get((r, c), [])
        )
        for r in range(rows)
    }

    # 综合评估候选花瓶的安全性得分
    def _vase_safety_score(cell: tuple[int, int]) -> tuple:
        r, c = cell
        has_shooter = lane_has_shooter.get(r, False)
        lane_zombie_count = len(lanes[r])
        # 排序键：
        # 1. 优先选择无僵尸且有射手的绝对安全行
        # 2. 列优先：越靠右得分越高 (c)
        # 3. 行僵尸数越少越优先 (-lane_zombie_count)
        # 4. 中间行偏好
        lane_safety = 2 if (has_shooter and lane_zombie_count == 0) else (1 if has_shooter else 0)
        center_bias = -abs(r - (rows // 2))
        return (lane_safety, c, -lane_zombie_count, center_bias)

    best_vase = max(candidate_vases, key=_vase_safety_score)
    br, bc = best_vase
    action_index = INTERACT_BASE + br * 9 + bc
    return (action_index, -1, br, bc, -1, f"4-5-smash-vase-r{br}c{bc}")
