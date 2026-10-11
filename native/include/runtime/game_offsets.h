#pragma once

#include <cstddef>
#include <cstdint>

namespace pvz::runtime {

// 游戏植物类型枚举 (参考游戏内部 ID)
enum PlantType : std::int32_t {
    Peashooter = 0,
    Sunflower = 1,
    CherryBomb = 2,
    Wallnut = 3,
    PotatoMine = 4,
    SnowPea = 5,
    Chomper = 6,
    Repeater = 7,
    Puffshroom = 8,
    Sunshroom = 9,
    Fumeshroom = 10,
    GraveBuster = 11,
    Hypnoshroom = 12,
    Scaredyshroom = 13,
    IceShroom = 14,
    DoomShroom = 15,
    LilyPad = 16,
    Squash = 17,
    Threepeater = 18,
    Tanglekelp = 19,
    Jalapeno = 20,
    Spikeweed = 21,
    Torchwood = 22,
    Tallnut = 23,
    Seashroom = 24,
    Plantern = 25,
    Cactus = 26,
    Blover = 27,
    SplitPea = 28,
    Starfruit = 29,
    Pumpkin = 30,
    Magnetshroom = 31,
    Cabbagepult = 32,
    FlowerPot = 33,
    Kernelpult = 34,
    CoffeeBean = 35,
    Garlic = 36,
    UmbrellaLeaf = 37,
    Marigold = 38,
    Melonpult = 39,
    GatlingPea = 40,
    TwinSunflower = 41,
    Gloomshroom = 42,
    Cattail = 43,
    WinterMelon = 44,
    GoldMagnet = 45,
    Spikerock = 46,
    CobCannon = 47,
    Imitater = 48
};

// 常见网格地块与障碍物 ID
enum GridItemType : std::int32_t {
    GridItemNone = 0,
    GridItemGrave = 1,
    GridItemCrater = 2,
    GridItemLadder = 3,
    GridItemPortal = 4
};

// 网格尺寸常量
constexpr int kGridRows = 6;
constexpr int kGridCols = 9;
constexpr int kMaxCardSlots = 10;

// 游戏逻辑帧时间常数 (1 tick = 10ms)
constexpr int kTickDurationMs = 10;
constexpr int kDefaultStepTicks = 20;

}  // namespace pvz::runtime
