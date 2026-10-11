#include "tutorial_skipper.h"

#include "pvz/api.h"
#include "../process/process_manager.h"

#include <atomic>
#include <cstdint>
#include <mutex>

namespace pvz::automation {
namespace {

// RL 关卡修复总开关：当前覆盖 1-1 和 1-2。
std::atomic_bool g_level_fixes_enabled{true};

// Board 偏移以经典版布局为基准；其它受支持版本按 sun 偏移差修正。
constexpr uintptr_t kClassicSunOffset = 0x5560;
constexpr uintptr_t kTutorialStateOffset = 0x5584;
constexpr uintptr_t kTutorialTimerOffset = 0x558C;
constexpr uintptr_t kZombieCountdownOffset = 0x559C;
constexpr uintptr_t kZombieCountdownStartOffset = 0x55A0;
constexpr uintptr_t kCurrentWaveOffset = 0x557C;

bool supported_version(PvzRuntime::VersionedPvZ& pvz) noexcept {
    return pvz.version_id() == PVZ_1_0_0_1051_EN ||
           pvz.version_id() == PVZ_1_2_0_1065_EN;
}

uintptr_t board_offset(PvzRuntime::VersionedPvZ& pvz, uintptr_t classic_offset) noexcept {
    const intptr_t delta = static_cast<intptr_t>(pvz.data().sun) -
                           static_cast<intptr_t>(kClassicSunOffset);
    return static_cast<uintptr_t>(static_cast<intptr_t>(classic_offset) + delta);
}

bool get_board_unlocked(PvzRuntime::VersionedPvZ& pvz, uintptr_t& board) {
    if (!pvz.GameOn() && !pvz.FindPvZ()) return false;
    if (!pvz.GameOn() || !supported_version(pvz)) return false;

    const int ui = pvz.GameUI();
    if (ui != 2 && ui != 3) return false;

    const uintptr_t lawn = pvz.ReadMemory<uintptr_t>({pvz.data().lawn});
    if (!lawn) return false;
    board = pvz.ReadMemory<uintptr_t>({lawn + pvz.data().board});
    return board != 0;
}

void write_state(PvzRuntime::VersionedPvZ& pvz, uintptr_t board, int state) {
    pvz.WriteMemory<int>({state}, {board + board_offset(pvz, kTutorialStateOffset)});
    pvz.WriteMemory<int>({-1}, {board + board_offset(pvz, kTutorialTimerOffset)});
}

}  // namespace

void TutorialSkipper::set_enabled(bool enable) noexcept {
    g_level_fixes_enabled.store(enable, std::memory_order_relaxed);
}

bool TutorialSkipper::enabled() noexcept {
    return g_level_fixes_enabled.load(std::memory_order_relaxed);
}

int TutorialSkipper::get_state() {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();

    uintptr_t board = 0;
    if (!get_board_unlocked(pvz, board)) return -1;
    return pvz.ReadMemory<int>({board + board_offset(pvz, kTutorialStateOffset)});
}

bool TutorialSkipper::skip_current() {
    if (!enabled()) return false;

    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();

    uintptr_t board = 0;
    if (!get_board_unlocked(pvz, board)) return false;

    const int level_id = pvz.ReadMemory<int>({board + pvz.data().adventure_level});
    if (level_id != 1 && level_id != 2) return false;

    const uintptr_t tutorial_state_addr =
        board + board_offset(pvz, kTutorialStateOffset);
    const int state = pvz.ReadMemory<int>({tutorial_state_addr});

    // 1-1：解除豌豆射手教学限制，并同步首波倒计时。
    if (level_id == 1) {
        switch (state) {
        case 1:  // LEVEL_1_PICK_UP_PEASHOOTER
        case 2:  // LEVEL_1_PLANT_PEASHOOTER
        case 3:  // LEVEL_1_REFRESH_PEASHOOTER
            pvz.WriteMemory<int>({99}, {board + board_offset(pvz, kZombieCountdownOffset)});
            pvz.WriteMemory<int>({99}, {board + board_offset(pvz, kZombieCountdownStartOffset)});
            write_state(pvz, board, 4);
            return true;
        default:
            return false;
        }
    }

    // 1-2：解除向日葵教学限制；首波未开始时同步原版倒计时语义。
    switch (state) {
    case 5:  // LEVEL_2_PICK_UP_SUNFLOWER
    case 6:  // LEVEL_2_PLANT_SUNFLOWER
    case 7:  // LEVEL_2_REFRESH_SUNFLOWER
        if (pvz.ReadMemory<int>({board + board_offset(pvz, kCurrentWaveOffset)}) == 0) {
            pvz.WriteMemory<int>({999}, {board + board_offset(pvz, kZombieCountdownOffset)});
            pvz.WriteMemory<int>({999}, {board + board_offset(pvz, kZombieCountdownStartOffset)});
        }
        write_state(pvz, board, 8);
        return true;
    default:
        return false;
    }
}

const char* TutorialSkipper::state_name(int state) noexcept {
    static const char* const names[] = {
        "OFF",
        "LEVEL_1_PICK_UP_PEASHOOTER",
        "LEVEL_1_PLANT_PEASHOOTER",
        "LEVEL_1_REFRESH_PEASHOOTER",
        "LEVEL_1_COMPLETED",
        "LEVEL_2_PICK_UP_SUNFLOWER",
        "LEVEL_2_PLANT_SUNFLOWER",
        "LEVEL_2_REFRESH_SUNFLOWER",
        "LEVEL_2_COMPLETED",
    };
    if (state < 0 || state >= static_cast<int>(sizeof(names) / sizeof(names[0]))) {
        return "UNKNOWN";
    }
    return names[state];
}

}  // namespace pvz::automation

extern "C" {

void PVZ_GYM_CALL pvz_gym_set_auto_skip_tutorial(bool enable) {
    pvz::automation::TutorialSkipper::set_enabled(enable);
}

bool PVZ_GYM_CALL pvz_gym_get_auto_skip_tutorial() {
    return pvz::automation::TutorialSkipper::enabled();
}

int PVZ_GYM_CALL pvz_gym_get_tutorial_state() {
    return pvz::automation::TutorialSkipper::get_state();
}

bool PVZ_GYM_CALL pvz_gym_skip_tutorial() {
    return pvz::automation::TutorialSkipper::skip_current();
}

}  // extern "C"
