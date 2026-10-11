// -----------------------------------------------------------------------------
// 自动化辅助模块 (Automation Helper) - 掉落物自动拾取
// 注意：本模块是测试与训练的便利工具，不属于 RL 动作空间，不参与任何策略决策与奖励计算。
// Automation helpers are training/testing conveniences and are not part of the RL agent action space.
// -----------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <unordered_set>

namespace pvz::automation {

// PvZ 内部 CoinType。这里列出自动化层需要区分的全部常见类型。
//
// 分类原则：
// 1. regular collectable: 阳光/货币，可持续自动收集；
// 2. terminal/progression reward: 过关或解锁奖励，需要自动领取以推进原版结算；
// 3. agent-owned collectable: 会变成策略动作资源，严禁自动收集。
enum class CollectableType : int {
    Silver = 1,
    Gold = 2,
    Diamond = 3,
    NormalSun = 4,
    SmallSun = 5,
    LargeSun = 6,

    FinalSeedPacket = 7,
    Trophy = 8,
    Shovel = 9,
    Almanac = 10,
    CarKeys = 11,
    Vase = 12,
    WateringCan = 13,
    Taco = 14,
    Note = 15,

    // 砸罐子、老虎机等模式掉落的可直接种植卡牌。
    // 玩家拾取后会进入 CURSOR_TYPE_PLANT_FROM_USABLE_COIN，属于 Agent 的动作资源，
    // 不能被普通 auto-collect 抢先拾取。
    UsableSeedPacket = 16,

    PresentPlant = 17,
    AwardMoneyBag = 18,
    AwardPresent = 19,
    AwardBagDiamond = 20,
    AwardSilverSunflower = 21,
    AwardGoldSunflower = 22,
    Chocolate = 23,
    AwardChocolate = 24,
    PresentMinigames = 25,
    PresentPuzzleMode = 26,
    PresentSurvivalMode = 27,
};

class AutoCollect {
public:
    // 执行一轮全图自动收集。
    // 仅当处于战斗状态 (UI == 3) 且 Board 有效时执行。
    // 默认收集普通资源和终局/进度奖励，但永远不会自动收 UsableSeedPacket。
    // 返回本次成功标记为“正在收集”的掉落物数量。
    static int collect_all();

    // 检查某种 CoinType 当前是否允许自动收集。
    // Agent-owned 类型即使被误加入白名单，也固定返回 false。
    static bool is_type_allowed(int coin_type);

    // 语义分类辅助。
    static bool is_regular_collectable(int coin_type);
    static bool is_terminal_reward(int coin_type);
    static bool is_agent_owned_collectable(int coin_type);

    // 配置允许/禁止收集的类型。
    // Agent-owned 类型不可被开启，以避免自动化吞掉策略动作资源。
    static void set_type_allowed(int coin_type, bool allowed);

    // 重置为默认白名单：普通资源 + 终局/进度奖励。
    static void reset_default_types();

private:
    static std::unordered_set<int>& allowed_types_set();
};

}  // namespace pvz::automation
