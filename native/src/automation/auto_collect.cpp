// -----------------------------------------------------------------------------
// 自动化辅助模块 (Automation Helper) - 掉落物自动拾取实现
// 注意：本模块是测试与训练的便利工具，不属于 RL 动作空间，不参与任何策略决策与奖励计算。
// Automation helpers are training/testing conveniences and are not part of the RL agent action space.
// -----------------------------------------------------------------------------
#include "auto_collect.h"
#include "../game/simulation_clock.h"
#include "../process/main_loop_barrier.h"
#include "../process/process_manager.h"
#include "../runtime/resident_executor.h"
#include "runtime/pvz_runtime_layout.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <vector>

namespace pvz::automation {

namespace {
std::mutex g_type_mutex;

void insert_default_collectables(std::unordered_set<int>& types) {
    // 普通资源：训练时持续自动收集。
    types.insert(static_cast<int>(CollectableType::Silver));
    types.insert(static_cast<int>(CollectableType::Gold));
    types.insert(static_cast<int>(CollectableType::Diamond));
    types.insert(static_cast<int>(CollectableType::NormalSun));
    types.insert(static_cast<int>(CollectableType::SmallSun));
    types.insert(static_cast<int>(CollectableType::LargeSun));

    // 关卡终局 / 进度奖励：若不领取，原版可能一直停留在奖励掉落阶段，
    // mLevelComplete 无法继续推进，导致下一次 reset/select_level 卡住。
    types.insert(static_cast<int>(CollectableType::FinalSeedPacket));
    types.insert(static_cast<int>(CollectableType::Trophy));
    // 一周目 1-4 的真实关卡奖励是 COIN_SHOVEL；它必须和其它进度奖励一样领取。
    types.insert(static_cast<int>(CollectableType::Shovel));
    types.insert(static_cast<int>(CollectableType::Almanac));
    types.insert(static_cast<int>(CollectableType::CarKeys));
    types.insert(static_cast<int>(CollectableType::Vase));
    types.insert(static_cast<int>(CollectableType::WateringCan));
    types.insert(static_cast<int>(CollectableType::Taco));
    types.insert(static_cast<int>(CollectableType::Note));
    types.insert(static_cast<int>(CollectableType::PresentPlant));
    types.insert(static_cast<int>(CollectableType::AwardMoneyBag));
    types.insert(static_cast<int>(CollectableType::AwardPresent));
    types.insert(static_cast<int>(CollectableType::AwardBagDiamond));
    types.insert(static_cast<int>(CollectableType::AwardSilverSunflower));
    types.insert(static_cast<int>(CollectableType::AwardGoldSunflower));
    types.insert(static_cast<int>(CollectableType::AwardChocolate));
    types.insert(static_cast<int>(CollectableType::PresentMinigames));
    types.insert(static_cast<int>(CollectableType::PresentPuzzleMode));
    types.insert(static_cast<int>(CollectableType::PresentSurvivalMode));

    // 明确不加入默认集合：
    // 16 UsableSeedPacket  -> Agent 动作资源，拾取后占用 Cursor，严禁 auto-collect。
    // 23 Chocolate         -> 普通特殊资源，暂不属于默认训练自动化范围。
}

int collect_coin_batch_native(
    PvzRuntime::VersionedPvZ& pvz,
    const std::vector<uintptr_t>& candidates) {
    if (candidates.empty()) return 0;

    const uintptr_t fn = PvzRuntime::coin_collect_address(pvz);
    if (!fn) return 0;

    // Persistent headless already owns and suspends the native main thread.
    // Acquiring another MainLoopBarrier in that state would see a non-zero
    // suspend count and fail. Reuse the existing ownership instead; outside
    // headless, retain the exact short barrier used by animated/normal modes.
    std::unique_ptr<process::MainLoopBarrier> local_barrier;
    if (!game::SimulationClock::headless_owns_main_loop() && !runtime::ResidentExecutor::instance().owner()) {
        local_barrier = std::make_unique<process::MainLoopBarrier>(pvz);
        if (!local_barrier->acquire()) return 0;
    }

    std::vector<uintptr_t> live;
    live.reserve(candidates.size());
    for (const uintptr_t coin : candidates) {
        if (!coin) continue;
        if (pvz.ReadMemory<bool>({coin + PvzRuntime::kCoinDeadOffset})) continue;
        if (pvz.ReadMemory<bool>({coin + PvzRuntime::kCoinBeingCollectedOffset})) continue;
        live.push_back(coin);
    }

    if (live.empty()) {
        if (local_barrier) local_barrier->release();
        return 0;
    }

    // 原版：__thiscall void Coin::Collect(); ECX = Coin*。
    // Keep the original function so all score/award/fade side effects remain
    // authoritative; only the synchronization and injection count are reduced.
    pvz.asm_init();
    for (const uintptr_t coin : live) {
        pvz.asm_mov_exx(Pt::Reg::ECX, static_cast<unsigned int>(coin));
        pvz.asm_call(static_cast<unsigned int>(fn));
    }
    pvz.asm_ret();
    pvz.asm_code_inject(true);

    if (local_barrier) local_barrier->release();
    return static_cast<int>(live.size());
}
}  // namespace

std::unordered_set<int>& AutoCollect::allowed_types_set() {
    static std::unordered_set<int> types = [] {
        std::unordered_set<int> defaults;
        insert_default_collectables(defaults);
        return defaults;
    }();
    return types;
}

bool AutoCollect::is_regular_collectable(int coin_type) {
    switch (coin_type) {
        case static_cast<int>(CollectableType::Silver):
        case static_cast<int>(CollectableType::Gold):
        case static_cast<int>(CollectableType::Diamond):
        case static_cast<int>(CollectableType::NormalSun):
        case static_cast<int>(CollectableType::SmallSun):
        case static_cast<int>(CollectableType::LargeSun):
            return true;
        default:
            return false;
    }
}

bool AutoCollect::is_terminal_reward(int coin_type) {
    switch (coin_type) {
        case static_cast<int>(CollectableType::FinalSeedPacket):
        case static_cast<int>(CollectableType::Trophy):
        case static_cast<int>(CollectableType::Shovel):
        case static_cast<int>(CollectableType::Almanac):
        case static_cast<int>(CollectableType::CarKeys):
        case static_cast<int>(CollectableType::Vase):
        case static_cast<int>(CollectableType::WateringCan):
        case static_cast<int>(CollectableType::Taco):
        case static_cast<int>(CollectableType::Note):
        case static_cast<int>(CollectableType::PresentPlant):
        case static_cast<int>(CollectableType::AwardMoneyBag):
        case static_cast<int>(CollectableType::AwardPresent):
        case static_cast<int>(CollectableType::AwardBagDiamond):
        case static_cast<int>(CollectableType::AwardSilverSunflower):
        case static_cast<int>(CollectableType::AwardGoldSunflower):
        case static_cast<int>(CollectableType::AwardChocolate):
        case static_cast<int>(CollectableType::PresentMinigames):
        case static_cast<int>(CollectableType::PresentPuzzleMode):
        case static_cast<int>(CollectableType::PresentSurvivalMode):
            return true;
        default:
            return false;
    }
}

bool AutoCollect::is_agent_owned_collectable(int coin_type) {
    // COIN_USABLE_SEED_PACKET is not a passive pickup. It becomes
    // CURSOR_TYPE_PLANT_FROM_USABLE_COIN and therefore represents a plant-placement
    // action resource that must remain under Agent control.
    return coin_type == static_cast<int>(CollectableType::UsableSeedPacket);
}

bool AutoCollect::is_type_allowed(int coin_type) {
    if (is_agent_owned_collectable(coin_type)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_type_mutex);
    const auto& s = allowed_types_set();
    return s.find(coin_type) != s.end();
}

void AutoCollect::set_type_allowed(int coin_type, bool allowed) {
    // 硬保护：掉落植物卡属于 Agent 动作资源，即使误配置也不允许自动拾取。
    if (is_agent_owned_collectable(coin_type)) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_type_mutex);
    auto& s = allowed_types_set();
    if (allowed) {
        s.insert(coin_type);
    } else {
        s.erase(coin_type);
    }
}

void AutoCollect::reset_default_types() {
    std::lock_guard<std::mutex> lock(g_type_mutex);
    auto& s = allowed_types_set();
    s.clear();
    insert_default_collectables(s);
}

int AutoCollect::collect_all() {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();

    // 奖励钱袋等关卡终局掉落出现时仍处于 SCENE_PLAYING (UI == 3)，
    // 因此这里必须继续在 UI == 3 时处理普通资源和 terminal reward。
    if (!pvz.GameOn()) return 0;
    if (pvz.GameUI() != 3) return 0;

    // 未知版本不猜 Coin::Collect 地址。宁可本轮不自动收，也不能错误远程 CALL。
    if (!PvzRuntime::coin_collect_address(pvz)) return 0;

    const uintptr_t board =
        pvz.ReadMemory<uintptr_t>({pvz.data().lawn, pvz.data().board});
    if (!board) return 0;

    const uintptr_t coins_field = board + PvzRuntime::coin_array_offset(pvz);
    const uintptr_t coins = pvz.ReadMemory<uintptr_t>({coins_field});
    int coin_max = pvz.ReadMemory<int>({coins_field + 0x04});
    coin_max = std::max(0, std::min(1024, coin_max));
    if (!coins || coin_max <= 0) return 0;

    std::vector<uintptr_t> regular;
    std::vector<uintptr_t> terminal;
    regular.reserve(static_cast<std::size_t>(coin_max));

    for (int i = 0; i < coin_max; ++i) {
        const uintptr_t coin =
            coins + static_cast<uintptr_t>(i) * PvzRuntime::kCoinStructSize;

        if (pvz.ReadMemory<bool>({coin + PvzRuntime::kCoinDeadOffset})) continue;
        if (pvz.ReadMemory<bool>({coin + PvzRuntime::kCoinBeingCollectedOffset})) continue;

        const int ctype = pvz.ReadMemory<int>({coin + 0x58});
        if (is_agent_owned_collectable(ctype)) continue;
        if (!is_type_allowed(ctype)) continue;

        if (is_terminal_reward(ctype)) {
            terminal.push_back(coin);
        } else {
            regular.push_back(coin);
        }
    }

    int collected_count = collect_coin_batch_native(pvz, regular);

    // Terminal rewards are rare and can start scene transitions. Keep them as
    // one-call batches so a transition caused by one reward cannot invalidate a
    // second Coin* inside the same remote program.
    for (const uintptr_t coin : terminal) {
        if (pvz.GameUI() != 3) break;
        collected_count += collect_coin_batch_native(
            pvz, std::vector<uintptr_t>{coin});
    }

    return collected_count;
}

}  // namespace pvz::automation
