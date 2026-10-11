"""Regenerate the reviewable teaching-deck analysis from the actual profiles."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from pvz_gym.adventure_curriculum import PLANT_REWARDS, ZOMBIE_NAMES, deck_profile
from pvz_gym.default_decks import PLANT_NAMES


PLANT_LESSONS = (
    "基础直线输出；屋顶斜坡前不宜作为主攻。",
    "先建立经济；夜间可用阳光菇替代，戴夫给向日葵则保留。",
    "局部范围爆发，处理重甲/集群；给应急留阳光。",
    "保护输出并争取时间；挡不住撑杆/海豚/跳跳的首次跳跃。",
    "廉价前期击杀，但需要埋设时间；不能当作撑杆必杀。",
    "一路一株减速，搭配其他输出；与火炬同一路会改变冰豌豆效果。",
    "快速吃掉铁桶等单体；消化期间需要坚果保护。",
    "提高持续火力，较贵；前期仍靠地雷/窝瓜稳经济。",
    "夜间免费开局，射程短；白天无咖啡豆只作补位。",
    "夜间便宜经济；成长前单次产量较小。",
    "范围穿透铁门护盾；白天需要咖啡豆。",
    "清墓扩大布阵空间，减少最终波从墓中出现的压力。",
    "让吃到它的僵尸反打；橄榄球可转化成友军火力。",
    "廉价远射，近敌会躲；需要前排缓冲。",
    "全场冻结，处理舞王集群、巨人和紧急节奏；白天需咖啡豆。",
    "大范围爆发但留下坑；别炸掉关键经济/载具布阵位置。",
    "泳池陆生植物的载具；选了载具还要有可放其上的输出。",
    "早期单体应急，也能击杀冰车；巨人需要另一张爆发配合。",
    "覆盖相邻三路，陆路对水路提供火力；选位置比堆数量更重要。",
    "水路单体击杀，对潜水/海豚有效；不能用于陆路。",
    "整行爆发并清冰道；屋顶投石车、巨人也可用作救场。",
    "陆路地面伤害并破轮胎；不作为屋顶/水路的冰车方案。",
    "增强豌豆输出，须配射手；不独立提供伤害。",
    "阻挡撑杆、海豚和跳跳；扶梯仍可让僵尸翻越。",
    "免费水路近程输出；夜间有效，白天会睡觉。",
    "固定照明关键区域；不直接处理气球。",
    "能打气球；4-3 首次气球时的已解锁解法。",
    "驱走气球并暂时吹散雾，常能替代仙人掌/路灯的部分槽位。",
    "向前及向后射击；4-6 首次矿工时的已解锁解法。",
    "多个方向覆盖，可处理后排威胁；需要正确位置，非单向主力替代。",
    "保护同格植物，不额外占种植格；注意小鬼落点和高坚果保护。",
    "吸走可吸附金属，如铁桶、门、工具、跳杆、梯子；白天需咖啡豆。",
    "投射跨屋顶斜坡；5-1 使用游戏给定初始花盆。",
    "屋顶扩建载具；5-1 通关后才获得种子卡。",
    "投射输出，黄油可提供短时控制；不能保证每发都控住敌人。",
    "白天唤醒蘑菇，是白天磁力菇/大喷菇/寒冰菇的依赖。",
    "引导能啃食它的步行僵尸改道；不能替代投石车/巨人专门应对。",
    "保护附近植物免受蹦极和投石；按覆盖范围布置，单株不保全场。",
    "产金币，几乎不贡献战斗能力；固定教学中只作后位候选/补位。",
    "强范围投射输出，适合最后常规屋顶关；昂贵，仍需前期经济/应急。",
)


def names(plants):
    return "、".join(PLANT_NAMES[p] for p in plants) or "—"


def main():
    lines = [
        "# 冒险一、二周目逐关教学选卡分析", "",
        "## 目录", "",
        "- [选择规则与解锁时机](#选择规则与解锁时机)",
        "- [40 张基础植物的教学用途](#40-张基础植物的教学用途)",
        "- [一周目逐关固定卡与候选](#一周目逐关固定卡与候选)",
        "- [二周目逐关固定卡与候选](#二周目逐关固定卡与候选)",
        "- [验证范围与来源](#验证范围与来源)", "",
        "## 选择规则与解锁时机", "",
        "Workbench 的 Adventure LIVE 与 Gym Training 卡组面板默认选择 **每关预设卡组**，配置值为 `fixed`。"
        "每关有固定的主卡顺序和候选顺序。实际僵尸预览用于提示缺口，不重排固定卡；"
        "用户仍可主动切换自动针对或编辑卡组。基础 EnvConfig 的默认值为 `native`，"
        "与 GUI 默认不同，详见 [选卡规则与入口](default_adventure_decks.md)。", "",
        "1. 一周目按本关开始前已获得的植物选择；本关通关奖励从下一关才能使用。"
        "即使晚期存档或修改后的游戏显示更多植物，也不把未来奖励混进一周目教学卡组。",
        "2. 先保留游戏不能撤下的戴夫卡，再依次选择主卡；缺卡或尚有空槽时按候选补齐。"
        "二周目可使用 40 张基础植物，但不假定升级卡或模仿者的购买状态。",
        "3. 六/八/十槽取固定顺序中实际可选的前六/八/十张。固定指选择规则固定；"
        "戴夫锁卡、真实解锁和实际槽位仍决定最终卡组。候选不是随机抽取。",
        "4. 3-1 在十槽环境中只有九张白天可用基础植物，固定加入小喷菇补位；"
        "它白天睡觉，不作为有效火力。教程早期如果真实可选卡总数不足槽位，明确报错，不解锁未来植物。",
        "5. 保龄球、打僵尸、砸罐子、传送带和僵王关依赖游戏预置/发牌。"
        "以下表中的配置用于记录教学用途；无交互选卡界面就跳过，不能理解为实际可自由挑十张卡。",
        "6. 白天磁力菇的固定选择要求咖啡豆先入组。戴夫锁入的睡眠蘑菇不会被删除，"
        "缺唤醒卡时会提示应对缺口。高坚果、火炬、保护伞等仍需要战斗策略正确布置。", "",
        "关键时机：3-3 用三线射手、水路前排与应急处理潜水，到 3-4 才有海草；"
        "3-6 先用辣椒/窝瓜处理冰车，到 3-7 才有地刺；3-8 海豚先靠海草与诱跳，到 3-9 才有高坚果；"
        "4-3 用仙人掌处理气球，4-4 才有三叶草；4-6 用裂荚处理矿工，4-9 才有磁力菇；"
        "5-1 靠初始花盆，5-2 才能选花盆种子；5-6 投石车先靠爆发，5-7 才有保护伞。", "",
        "## 40 张基础植物的教学用途", "",
        "| 植物 | 首次可用关卡 | 教学用途、依赖和限制 |", "| --- | --- | --- |",
    ]
    unlock_at = {0: 1, **{p: level + 1 for level, p in PLANT_REWARDS.items()}}
    for plant, lesson in enumerate(PLANT_LESSONS):
        level = unlock_at[plant]
        lines.append(f"| {PLANT_NAMES[plant]} | {(level-1)//10+1}-{(level-1)%10+1} | {lesson} |")
    for playthrough, title in ((1, "一周目"), (2, "二周目")):
        lines += ["", f"## {title}逐关固定卡与候选", ""]
        if playthrough == 2:
            lines += ["二周目植物列表示该关复习的教学卡，不表示重新解锁。"
                      "戴夫锁卡先占槽，表中主卡和候选再补剩余槽位。"
                      "二周目实际僵尸阵容以游戏为准，预览中出现额外威胁会给缺口提示。", ""]
        for area, scene in enumerate(("白天草地", "夜晚", "泳池", "雾夜", "屋顶"), 1):
            lines += [f"### {title} {area}：{scene}", "",
                      "| 关卡 | 新可用 / 通关获卡 | 新僵尸或关型 | 固定主卡（从左到右） | 候选（从左到右） | 教学与应对 |",
                      "| --- | --- | --- | --- | --- | --- |"]
            for level in range((area-1)*10+1, area*10+1):
                p = deck_profile(level, playthrough)
                award = PLANT_REWARDS.get(level)
                new = names(p.new_plants)
                reward = names((award,)) if award is not None else "无新卡"
                threats = "、".join(ZOMBIE_NAMES[z] for z in p.new_zombies) or "复习已有威胁"
                if p.special:
                    threats += f"；{p.special}"
                lines.append(f"| {area}-{(level-1)%10+1} | {new} / {reward} | {threats} | "
                             f"{names(p.primary)} | {names(p.candidates)} | {p.lesson} |")
            lines.append("")
    lines += [
        "## 验证范围与来源", "",
        "这是项目的教学选卡设计，选卡顺序与候选可在 "
        "[python/pvz_gym/adventure_curriculum.py](../python/pvz_gym/adventure_curriculum.py) 修改。"
        "通过 `python -m scripts.export_adventure_curriculum` 同步此文档。"
        "卡组配置覆盖 50 关 × 2 周目；覆盖配置不代表脚本已通关所有关卡。", "",
        "解锁顺序核对 [LawnApp::GetAwardSeedForLevel / GetSeedsAvailable](https://github.com/Patoke/re-plants-vs-zombies/blob/main/LawnApp.cpp)。"
        "新僵尸时机核对 [gZombieDefs](https://github.com/Patoke/re-plants-vs-zombies/blob/main/Lawn/Zombie.cpp)。"
        "每关可出僵尸仍受 [Board::CanZombieSpawnOnLevel](https://github.com/Patoke/re-plants-vs-zombies/blob/main/Lawn/Board.cpp) 限制；"
        "不能把‘此前已经介绍过’理解为该关必然出现。", "",
        "运行时以原生 `SeedTypeAvailable`、实际选卡状态、动作掩码和戴夫锁卡为准。"
        "单元测试覆盖逐关可用卡与候选、六/八/十槽、固定卡不随预览变化、戴夫锁卡和同局训练衔接。",
    ]
    target = Path(__file__).resolve().parents[1] / "docs" / "adventure_curriculum.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
