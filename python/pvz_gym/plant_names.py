"""Chinese display names indexed by native plant type (before observation encoding)."""

PLANT_NAMES_ZH = (
    "豌豆射手", "向日葵", "樱桃炸弹", "坚果", "土豆雷",
    "寒冰射手", "大嘴花", "双发射手", "小喷菇", "阳光菇",
    "大喷菇", "墓碑吞噬者", "魅惑菇", "胆小菇", "寒冰菇",
    "毁灭菇", "睡莲", "窝瓜", "三线射手", "缠绕海草",
    "火爆辣椒", "地刺", "火炬树桩", "高坚果", "海蘑菇",
    "路灯花", "仙人掌", "三叶草", "裂荚射手", "杨桃",
    "南瓜头", "磁力菇", "卷心菜投手", "花盆", "玉米投手",
    "咖啡豆", "大蒜", "叶子保护伞", "金盏花", "西瓜投手",
    "机枪射手", "双子向日葵", "忧郁菇", "香蒲", "冰瓜",
    "吸金磁", "地刺王", "玉米加农炮", "模仿者", "爆炸坚果", "巨型坚果",
)


def plant_name_zh(plant_type: int) -> str:
    if 0 <= plant_type < len(PLANT_NAMES_ZH):
        return PLANT_NAMES_ZH[plant_type]
    return f"植物{plant_type}" if plant_type >= 0 else "未知卡牌"
