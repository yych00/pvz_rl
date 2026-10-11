#include "runtime.h"
#include "main_loop_barrier.h"
#include "level_reset_code.h"
#include "process_manager.h"
#include "../runtime/resident_executor.h"
#include "../runtime/turbo_runtime.h"
#include "../runtime/native_runtime.h"

#include "../game/simulation_clock.h"
#include "pvz/types.h"
#include "runtime/compat_x64.h"
#include "runtime/pvz_runtime_layout.h"

#include <cstdint>
#include <memory>
#include <mutex>

namespace {
constexpr std::uintptr_t kPlayerFinishedAdventureOffset = 0x2C;

int trigger_native_enter_level(PvzRuntime::VersionedPvZ& pvz) {
    uintptr_t fn = PvzRuntime::pre_new_game_address(pvz);
    if (fn == 0) return static_cast<int>(pvz::ActionError::UnsupportedGameVersion);

    HWND hwnd = pvz.game_window();
    if (hwnd && ::IsWindow(hwnd)) {
        // 若窗口被最小化，必须在挂起主线程屏障前恢复显示，确保主线程消息循环处于活动就绪状态，
        // 绝不能在挂起线程后再调用 ShowWindow 导致死锁，同时静默恢复不抢占前台焦点
        if (::IsIconic(hwnd)) {
            ::ShowWindowAsync(hwnd, SW_SHOWNOACTIVATE);
            ::Sleep(30);
        }
    }

    std::unique_ptr<pvz::process::MainLoopBarrier> barrier;
    if (!pvz::game::SimulationClock::headless_owns_main_loop() && !pvz::runtime::ResidentExecutor::instance().owner()) {
        barrier = std::make_unique<pvz::process::MainLoopBarrier>(pvz);
        if (!barrier->acquire()) return static_cast<int>(pvz::ActionError::ActionTimeout);
    }

    pvz::process::build_level_reset_code(pvz, pvz.data().lawn, fn,
        PvzRuntime::process_safe_delete_list_address(pvz),
        PvzRuntime::kill_dialog_address(pvz),
        pvz.version_id() == PVZ_1_2_0_1065_EN ? 0x0044fb20 : 0,
        pvz::runtime::ResidentExecutor::instance().owner() ? 0x0044fc30 : 0);
    DWORD exit_code = STILL_ACTIVE;
    bool completed;
    if (pvz::runtime::ResidentExecutor::instance().owns(pvz)) {
        pvz.asm_code_inject(); exit_code = 0; completed = true;
    } else completed = pvz.Code::asm_code_inject(pvz.process_handle(), 5000, &exit_code);
    // On failure RAII resumes the window thread, including an exception path
    // waiting for ShowWindow. The caller stops instead of issuing more resets.
    return completed && exit_code == 0 ? 0 : static_cast<int>(pvz::ActionError::ActionTimeout);
}

// AwardScreen::KeyChar(' ') -> StartButtonPressed() is the original game path.
// That path calls LawnApp::KillAwardScreen() before PreNewGame(), so both the
// widget and LawnApp::mAwardScreen are cleaned up correctly.  Do not zero the
// AwardScreen pointer manually and do not call PreNewGame directly from UI=5.
bool trigger_award_continue(PvzRuntime::VersionedPvZ& pvz) {
    HWND hwnd = pvz.game_window();
    if (!hwnd || !::IsWindow(hwnd)) return false;

    // 静默恢复最小化窗口，不夺取前台焦点
    if (::IsIconic(hwnd)) ::ShowWindowAsync(hwnd, SW_SHOWNOACTIVATE);

    // Send WM_CHAR directly because AwardScreen handles KeyChar, not merely a
    // raw key-down.  The Python reset lifecycle already polls until UI=5 is
    // actually left, so this function intentionally stays asynchronous.
    return ::PostMessageW(hwnd, WM_CHAR, static_cast<WPARAM>(L' '), 0x10000000) != FALSE;
}

bool can_native_enter_level_from_ui(int ui) noexcept {
    switch (ui) {
        case 2:  // SCENE_LEVEL_INTRO / seed chooser / dialogue
        case 3:  // SCENE_PLAYING
        case 4:  // SCENE_ZOMBIES_WON -- terminal loss/cutscene/game-over dialog
            return true;
        default:
            return false;
    }
}
}

namespace pvz::process {

bool attach() {
    return ProcessManager::instance().attach();
}

bool game_on() {
    return ProcessManager::instance().is_game_on();
}

int ui() {
    return ProcessManager::instance().get_ui();
}

int set_adventure_playthrough(int playthrough) {
    if (playthrough != 1 && playthrough != 2) return 1002;

    auto& mgr = ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return 1001;
    if (!pvz.GameOn()) return 1001;

    // Gym 的“一周目/二周目”是单关 episode 配置，不要求真的把整套
    // Adventure 从 1-1 通到 5-10。原版 IsFirstTimeAdventureMode()/
    // HasFinishedAdventure() 最终读取 PlayerInfo::mFinishedAdventure，
    // 因此每个 episode 启动前直接把该字段规范化为 0/1 即可：
    //   0 -> 一周目规则/教学/早期卡组
    //   1 -> 二周目及以后规则/重玩选卡
    // 下一局会再次显式写入，所以 FIRST_RUN x5 与 REPLAY x5 可以分别
    // 连续运行，而不会因为上一局通关就自动切换模式。
    const int finished_adventure = (playthrough == 1) ? 0 : 1;
    pvz.WriteMemory<int>(finished_adventure, {
        pvz.data().lawn,
        pvz.data().user_data,
        kPlayerFinishedAdventureOffset,
    });
    return 0;
}

int select_level(int level_id) {
    if (level_id < 1 || level_id > 50) return 1002;
    // Stop the UI-service worker before taking its shared mutex. A reset must
    // own a local barrier so every timeout releases the game window thread.
    if (!game::SimulationClock::end_headless_mode()) return 1001;
    auto& mgr = ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return 1001;
    if (!pvz.GameOn()) return 1001;

    pvz.ensure_remote_idle();
    const int ui = pvz.GameUI();
    const uintptr_t board = pvz.ReadMemory<uintptr_t>({pvz.data().lawn, pvz.data().board});

    // Deterministic E2E/Gym stepping pauses Board::Update while UI=3.  A run
    // may then finish or be replaced while another client still owns that
    // pause.  Terminal transitions must never inherit a frozen Board.
    if ((ui == 4 || ui == 5) && board != 0) {
        pvz.WriteMemory<bool>(false, {board + pvz.data().game_paused});
    }

    // UI=5 是一个独立 AwardScreen widget，不属于 Board。终局掉落物被
    // auto_collect 拾取后，原版会正常进入这里。若此时直接调用 PreNewGame，
    // KillBoard 不会删除 AwardScreen，于是新关卡虽然已经创建，旧奖励界面
    // 仍覆盖在上面。必须先走原版 AwardScreen::KeyChar -> StartButtonPressed，
    // 让它自己 KillAwardScreen，再由原版继续进入 PreNewGame。
    //
    // 该消息是异步的；PvZClient::_select_level_with_terminal_retry 会等待
    // phase 离开 AWARD，并在必要时重试，因此这里不要 sleep，也不要手动
    // 清空 LawnApp+0x778 指针。
    //
    // 注意：必须在发空格键之前预写入目标 level_id。原版 AwardScreen 关闭后
    // 会直接调用 PreNewGame，读取的是 PlayerData::mCurrentLevel（通关后已
    // 自动 +1），不预写则游戏会进入下一关而非 Gym 请求的目标关卡。
    if (ui == 5 && !runtime::ResidentExecutor::instance().owner()) {
        mgr.apply_crash_prevention_patches_nolock();
        pvz.WriteMemory<int>(level_id, {pvz.data().lawn, pvz.data().user_data, pvz.data().level});
        pvz.WriteMemory<int>(0, {pvz.data().lawn, pvz.data().game_mode});
        if (board != 0) {
            pvz.WriteMemory<int>(level_id, {board + pvz.data().adventure_level});
        }
        return trigger_award_continue(pvz) ? 0 : static_cast<int>(ActionError::ActionTimeout);
    }

    // 允许从以下具有有效 LawnApp/Board 生命周期的场景直接重开：
    //   UI=2 SCENE_LEVEL_INTRO
    //   UI=3 SCENE_PLAYING
    //   UI=4 SCENE_ZOMBIES_WON
    //
    // UI=4 是 Gym 的关键路径：僵尸进屋后原版会停留在失败 CutScene，随后还要
    // 等 GameOverDialog/点击才能 EndLevel。RL reset 不应等待这段纯 UI 结算，
    // 直接调用 PreNewGame 让原版 KillBoard 并创建新 Board 即可。
    //
    // 主菜单(UI=1)等界面的 GameSelector/渲染主线程状态不同，仍维持安全保护，
    // 只预写目标关卡号，不在这些界面强行注入 PreNewGame。
    // 1. 预写入目标关卡号至用户存档与 Board 结构，并确保防崩补丁常驻。
    mgr.apply_crash_prevention_patches_nolock();
    pvz.WriteMemory<int>(level_id, {pvz.data().lawn, pvz.data().user_data, pvz.data().level});
    pvz.WriteMemory<int>(0, {pvz.data().lawn, pvz.data().game_mode});
    if (board != 0) {
        pvz.WriteMemory<int>(level_id, {board + pvz.data().adventure_level});
    }

    if (!can_native_enter_level_from_ui(ui)) {
        if ((ui == 1 || ui == 5) && (pvz::runtime::TurboRuntime::instance().active() ||
            pvz::runtime::NativeRuntime::instance().options().executor == pvz::runtime::ExecutorKind::Resident))
            return trigger_native_enter_level(pvz);
        return 0;
    }

    // 2. 原生引擎级 CALL 注入秒切目标关卡。
    return trigger_native_enter_level(pvz);
}

int current_level() {
    auto& mgr = ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return 0;
    if (!pvz.GameOn()) return 0;

    const int current_ui = pvz.GameUI();
    int level = 0;
    // SCENE_ZOMBIES_WON / SCENE_AWARD 期间旧 Board 仍然存在；优先返回它的
    // 真实关卡号，便于终局日志/reset 判断，而不是过早退回 user_data。
    if (current_ui == 2 || current_ui == 3 || current_ui == 4 || current_ui == 5) {
        const uintptr_t board = pvz.ReadMemory<uintptr_t>({pvz.data().lawn, pvz.data().board});
        if (board != 0) level = pvz.ReadMemory<int>({board + pvz.data().adventure_level});
    }
    if (level <= 0) level = pvz.ReadMemory<int>({pvz.data().lawn, pvz.data().user_data, pvz.data().level});
    return level > 0 ? level : 1;
}

}  // namespace pvz::process
