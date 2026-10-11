#include "simulation_clock.h"
#include "board_update_batch.h"
#include "../process/main_loop_barrier.h"
#include "../process/process_manager.h"
#include "../runtime/resident_executor.h"

#include "runtime/pvz_runtime_layout.h"

#include <chrono>
#include <algorithm>
#include <memory>
#include <thread>

namespace pvz::game {

// ────────────────────────────────────────────────────────────────────────────
// 内部辅助（无锁版本，调用方必须已持有 mgr.mutex()）
// ────────────────────────────────────────────────────────────────────────────

static uintptr_t board_ptr_nolock(PvzRuntime::VersionedPvZ& pvz) noexcept {
    return pvz.ReadMemory<uintptr_t>({pvz.data().lawn, pvz.data().board});
}

static std::int64_t get_clock_nolock(PvzRuntime::VersionedPvZ& pvz) noexcept {
    uintptr_t board = board_ptr_nolock(pvz);
    if (!board) return 0;
    return static_cast<std::int64_t>(pvz.ReadMemory<int>({board + pvz.data().game_clock}));
}

// A paused Board should only be restored while the original battle is still
// SCENE_PLAYING. Board::Update may transition to ZOMBIES_WON/AWARD inside the
// injected tick; re-applying mPaused after that transition freezes the terminal
// screen and prevents the original UI from finishing its animation/state flow.
static bool should_restore_battle_pause(
    PvzRuntime::VersionedPvZ& pvz, uintptr_t original_board) noexcept {
    return original_board != 0 && pvz.GameUI() == 3 &&
           board_ptr_nolock(pvz) == original_board;
}

// 批量推进 N 个 ticks 注入（远程循环一次执行完毕，耗时仅数毫秒，实现瞬间瞬移）
static int fast_advance_ticks_nolock(
    PvzRuntime::VersionedPvZ& pvz, uintptr_t board, int ticks, uintptr_t fn,
    std::uint8_t scene, bool count_updates = false) {
    if (ticks <= 0 || !board || fn == 0) return -1;
    if (Pt::Code::has_pending_remote_call(pvz.process_handle())) return -1;

    std::unique_ptr<process::MainLoopBarrier> barrier;
    if (!SimulationClock::headless_owns_main_loop() && !runtime::ResidentExecutor::instance().owner()) {
        barrier = std::make_unique<process::MainLoopBarrier>(pvz);
        if (!barrier->acquire()) return -1;
    }
    // The original Board may have disappeared while acquiring the barrier.
    if (pvz.GameUI() != scene || board_ptr_nolock(pvz) != board) return 0;

    const bool currently_paused = pvz.ReadMemory<bool>({board + pvz.data().game_paused});
    const auto start_clock = get_clock_nolock(pvz);
    // CutScene::Update deliberately waits for the first Board draw. A resident
    // owner never draws: satisfy only that initialization gate, leaving preload,
    // effects, animation clocks and the original cutscene logic intact. Offset
    // verified in both CutScene::Update and Board::Draw in 1.2.0.1065 EN.
    if (scene == 2 && runtime::ResidentExecutor::instance().owner()) {
        if (pvz.version_id() != PVZ_1_2_0_1065_EN) return -1;
        if (pvz.ReadMemory<int>({board + 0x5570}) == 0)
            pvz.WriteMemory<int>(1, {board + 0x5570});
    }
    pvz.WriteMemory<bool>(false, {board + pvz.data().game_paused});
    const int issued = inject_board_update_batch_nolock(pvz, board, ticks, fn, scene);

    // Restore before returning ownership to the main thread. Never write to
    // the original address after Board replacement, or pause a terminal scene.
    if (currently_paused && should_restore_battle_pause(pvz, board)) {
        pvz.WriteMemory<bool>(true, {board + pvz.data().game_paused});
    }
    if (issued < 0) return -1;
    if (count_updates || board_ptr_nolock(pvz) != board) return issued;
    // Sample under the barrier, before the native thread can destroy Board.
    return static_cast<int>(std::max<std::int64_t>(0, get_clock_nolock(pvz) - start_clock));
}

static bool step_one_tick_nolock(PvzRuntime::VersionedPvZ& pvz) {
    if (pvz.GameUI() != 3) return false;
    const auto board = board_ptr_nolock(pvz);
    return fast_advance_ticks_nolock(
        pvz, board, 1, PvzRuntime::board_update_address(pvz), 3, true) == 1;
}

// ────────────────────────────────────────────────────────────────────────────
// pause / resume / is_paused
// 设置 Board::mPaused 标志。
// ────────────────────────────────────────────────────────────────────────────
bool SimulationClock::pause() {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return false;
    if (pvz.GameUI() != 3) return false;

    uintptr_t board = board_ptr_nolock(pvz);
    if (!board) return false;

    pvz.WriteMemory<bool>(true, {board + pvz.data().game_paused});
    return true;
}

bool SimulationClock::resume() {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return false;

    // Cleanup must also work after Board::Update changed LawnApp::mGameScene
    // from SCENE_PLAYING (3) to ZOMBIES_WON (4) or AWARD (5).  The Board is
    // still authoritative here, so do not reject resume merely because the
    // semantic scene is already terminal.
    uintptr_t board = board_ptr_nolock(pvz);
    if (!board) return false;

    pvz.WriteMemory<bool>(false, {board + pvz.data().game_paused});
    return true;
}

bool SimulationClock::is_paused() {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return false;

    // Same cleanup rule as resume(): terminal scenes may still own the Board
    // whose mPaused flag was set by the simulation controller.
    uintptr_t board = board_ptr_nolock(pvz);
    if (!board) return false;

    return pvz.ReadMemory<bool>({board + pvz.data().game_paused});
}

// ────────────────────────────────────────────────────────────────────────────
// get_game_clock
// ────────────────────────────────────────────────────────────────────────────
std::int64_t SimulationClock::get_game_clock() {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return 0;
    if (pvz.GameUI() != 3) return 0;
    return get_clock_nolock(pvz);
}

// ────────────────────────────────────────────────────────────────────────────
// step_one_tick（公有接口：加锁后委托无锁版本）
// ────────────────────────────────────────────────────────────────────────────
bool SimulationClock::step_one_tick() {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return false;
    return step_one_tick_nolock(pvz);
}

// ────────────────────────────────────────────────────────────────────────────
// advance_ticks(n)
//
// 瞬间快进 n 个逻辑 tick，以 mMainCounter 实际变化量为判据。
// 优先使用超高速批量注入模式，单次调用即可瞬间完成数百帧推演（僵尸瞬移效果）；
// 若注入条件不满足则安全回退至自然时间等待模式。
// ────────────────────────────────────────────────────────────────────────────
int SimulationClock::advance_ticks(int ticks) {
    if (ticks <= 0) return 0;

    auto& mgr = process::ProcessManager::instance();
    if (!mgr.is_battle_ready()) return -1;
    mgr.set_background_running(true);

    // 持锁一次，内部全程无锁
    std::unique_lock<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return -1;

    uintptr_t board = board_ptr_nolock(pvz);
    if (!board) return -1;

    const uintptr_t fn = PvzRuntime::board_update_address(pvz);
    if (fn != 0) {
        // ── 超高速注入快进模式：远程批量循环调用 Board::Update() ─────────────
        // A failed safety handoff must not fall through to an unprotected path.
        return fast_advance_ticks_nolock(pvz, board, ticks, fn, 3);
    }

    // ── 回退模式：实时等待游戏自然运行 ──────────────────────────────────────
    const bool was_paused = pvz.ReadMemory<bool>({board + pvz.data().game_paused});
    if (was_paused) {
        pvz.WriteMemory<bool>(false, {board + pvz.data().game_paused});
    }

    const std::int64_t start = get_clock_nolock(pvz);
    const std::int64_t target = start + static_cast<std::int64_t>(ticks);
    const int timeout_ms = ticks * 30 + 1000;
    const auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::milliseconds(timeout_ms);
    std::int64_t cur = start;
    while (std::chrono::steady_clock::now() < deadline && cur < target) {
        if (!should_restore_battle_pause(pvz, board)) break;
        // 短暂解锁让游戏主线程推进
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        lock.lock();
        if (board_ptr_nolock(pvz) != board) break;
        cur = get_clock_nolock(pvz);
    }

    // 若调用前处于暂停状态，仅在仍处于战斗时恢复暂停；终局必须继续运行。
    if (was_paused && should_restore_battle_pause(pvz, board)) {
        pvz.WriteMemory<bool>(true, {board + pvz.data().game_paused});
    }

    return static_cast<int>(cur - start);
}

// ────────────────────────────────────────────────────────────────────────────
// advance_intro_ticks(n)
//
// 关卡开场不是 battle-ready，mMainCounter 也不是可靠的推进判据；但原版
// Board::Update() 会调用 CutScene::Update()，而 mCutsceneTime 每次 Update 按原版
// 逻辑递增。因此 prebattle automation 直接按 Board::Update 次数推进，不等待现实时间。
// ────────────────────────────────────────────────────────────────────────────
int SimulationClock::advance_intro_ticks(int ticks) {
    if (ticks <= 0) return 0;

    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return -1;

    // 只允许用于原版 SCENE_LEVEL_INTRO，避免把该接口误用于菜单/战斗。
    if (pvz.GameUI() != 2) return -1;

    const uintptr_t board = board_ptr_nolock(pvz);
    if (!board) return -1;

    const uintptr_t fn = PvzRuntime::board_update_address(pvz);
    if (fn == 0) return -1;

    return fast_advance_ticks_nolock(pvz, board, ticks, fn, 2, true);
}

int SimulationClock::advance_post_seed_intro() {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() || pvz.version_id() != PVZ_1_2_0_1065_EN ||
        pvz.GameUI() != 2) return 0;
    const auto board = board_ptr_nolock(pvz);
    if (!board) return 0;
    const auto cut_scene = pvz.read_memory_checked<uintptr_t>({board + pvz.data().cut_scene});
    if (!cut_scene || pvz.read_memory_checked<bool>({cut_scene + 0x2c})) return 0;
    const auto time = static_cast<std::int64_t>(pvz.read_memory_checked<int>({cut_scene + 0x08}));
    const auto dave_time = static_cast<std::int64_t>(pvz.read_memory_checked<int>({cut_scene + 0x20}));
    if (time < dave_time + 4250 || time >= dave_time + 4510) return 0;
    const auto fn = PvzRuntime::board_update_address(pvz);
    if (!fn) return -1;
    // Run every original update, including exactly DaveTime + 4500. LIVE
    // acquires the main-loop barrier through fast_advance_ticks_nolock;
    // directly assigning the target time would skip InitLawnMowers again.
    // CutScene::Update adds 10 to mCutsceneTime per update (0x43c378).
    constexpr int kTimePerUpdate = 10;
    return fast_advance_ticks_nolock(pvz, board,
        static_cast<int>((dave_time + 4510 - time + kTimePerUpdate - 1) / kTimePerUpdate),
        fn, 2, true);
}

int SimulationClock::advance_terminal_updates(int updates) {
    if (updates <= 0) return 0;
    auto& manager = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(manager.mutex());
    auto& pvz = manager.runtime();
    if (!pvz.GameOn() || pvz.GameUI() != 4) return -1;
    const uintptr_t board = board_ptr_nolock(pvz);
    if (!board) return -1;
    const uintptr_t fn = PvzRuntime::board_update_address(pvz);
    if (fn == 0) return -1;

    // 终局失败过场限制单批次推进规模，并在每批推进后校验场景与 Board 是否依然有效，
    // 避免一口气盲目推进上千次 Updates 导致 Board 状态机越界或堆析构损坏。
    int remaining = std::min(updates, 200);
    int total_issued = 0;
    while (remaining > 0) {
        if (!pvz.GameOn() || pvz.GameUI() != 4 || board_ptr_nolock(pvz) != board) break;
        const int batch = std::min(remaining, 64);
        const int issued = fast_advance_ticks_nolock(pvz, board, batch, fn, 4, true);
        if (issued <= 0) break;
        total_issued += issued;
        remaining -= issued;
        if (issued < batch) break;
    }
    return total_issued;
}

}  // namespace pvz::game
