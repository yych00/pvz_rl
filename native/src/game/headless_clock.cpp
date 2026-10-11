#include "simulation_clock.h"
#include "board_update_batch.h"
#include "../process/main_loop_barrier.h"
#include "../process/process_manager.h"

#include "runtime/pvz_runtime_layout.h"

#include <Windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

namespace pvz::game {
namespace {

std::atomic<bool> g_headless_active{false};
bool g_restore_paused = false;
int g_restore_frame_duration = 0;
uintptr_t g_headless_board = 0;
DWORD g_headless_pid = 0;
std::unique_ptr<process::MainLoopBarrier> g_headless_barrier;
std::atomic<bool> g_headless_owns_main_loop{false};
std::atomic<std::uint64_t> g_ui_service_count{0};
std::atomic<std::uint64_t> g_ui_service_failures{0};
std::atomic<bool> g_ui_service_worker_running{false};
std::chrono::steady_clock::time_point g_last_ui_service{};

std::mutex g_ui_service_worker_mutex;
HANDLE g_ui_service_stop_event = nullptr;
HANDLE g_ui_service_thread = nullptr;

constexpr int kMaxInjectedBatchTicks = 64;

// 训练时为速度长期独占主线程，但界面服务不能依赖 Python 再次调用 step。
// 原生看门狗会独立释放暂停中的窗口线程，周期控制在 Windows 无响应阈值以内，
// 同步开销约每几秒支付一次，而不是每个决策区间都支付。
constexpr int kUiServiceIntervalMs = 3000;
constexpr int kUiServiceWindowMs = 20;
constexpr int kHeadlessFrameDurationMs = 1;
constexpr int kBarrierAcquireAttempts = 3;
constexpr int kBarrierRetryDelayMs = 2;

struct HeadlessDiagnostics {
    SimulationClock::HeadlessBeginStatus status =
        SimulationClock::HeadlessBeginStatus::Ok;
    int ui_before = -1;
    int ui_after = -1;
    uintptr_t board_before = 0;
    uintptr_t board_after = 0;
    std::int64_t clock_before = 0;
    std::int64_t clock_after = 0;
    bool paused_before = false;
    bool paused_after = false;
};

std::mutex g_diag_mutex;
HeadlessDiagnostics g_diag;

const char* begin_status_name(
    SimulationClock::HeadlessBeginStatus status) noexcept {
    using S = SimulationClock::HeadlessBeginStatus;
    switch (status) {
    case S::Ok: return "HEADLESS_OK";
    case S::AlreadyActive: return "ALREADY_ACTIVE";
    case S::GameNotFound: return "GAME_NOT_FOUND";
    case S::NotInBattle: return "NOT_IN_BATTLE";
    case S::BoardNull: return "BOARD_NULL";
    case S::UpdateFnNull: return "UPDATE_FN_NULL";
    case S::BattleChanged: return "BATTLE_CHANGED";
    case S::BarrierAcquireFailed: return "BARRIER_ACQUIRE_FAILED";
    case S::UiWorkerStartFailed: return "UI_WORKER_START_FAILED";
    }
    return "UNKNOWN";
}

void set_diagnostics(const HeadlessDiagnostics& value) noexcept {
    std::lock_guard<std::mutex> lock(g_diag_mutex);
    g_diag = value;
}

uintptr_t board_ptr_nolock(PvzRuntime::VersionedPvZ& pvz) noexcept {
    return pvz.ReadMemory<uintptr_t>({pvz.data().lawn, pvz.data().board});
}

std::int64_t clock_for_board_nolock(
    PvzRuntime::VersionedPvZ& pvz,
    uintptr_t board) noexcept {
    if (!board) return 0;
    return static_cast<std::int64_t>(
        pvz.ReadMemory<int>({board + pvz.data().game_clock}));
}

bool is_same_battle_nolock(
    PvzRuntime::VersionedPvZ& pvz,
    uintptr_t board) noexcept {
    return pvz.GameUI() == 3 && board != 0 && board_ptr_nolock(pvz) == board;
}

void release_headless_barrier_nolock() noexcept {
    g_headless_owns_main_loop.store(false, std::memory_order_release);
    if (g_headless_barrier) {
        g_headless_barrier->release();
        g_headless_barrier.reset();
    }
}

bool acquire_headless_barrier_nolock(PvzRuntime::VersionedPvZ& pvz) {
    if (g_headless_barrier && g_headless_barrier->acquired()) {
        g_headless_owns_main_loop.store(true, std::memory_order_release);
        return true;
    }

    release_headless_barrier_nolock();
    for (int attempt = 0; attempt < kBarrierAcquireAttempts; ++attempt) {
        auto candidate = std::make_unique<process::MainLoopBarrier>(pvz);
        if (candidate->acquire()) {
            g_headless_barrier = std::move(candidate);
            g_headless_owns_main_loop.store(true, std::memory_order_release);
            return true;
        }
        if (attempt + 1 < kBarrierAcquireAttempts) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(kBarrierRetryDelayMs));
        }
    }
    return false;
}

void request_ui_service_worker_stop() noexcept {
    std::lock_guard<std::mutex> lock(g_ui_service_worker_mutex);
    if (g_ui_service_stop_event) {
        ::SetEvent(g_ui_service_stop_event);
    }
}

void restore_headless_frame_duration_nolock(
    PvzRuntime::VersionedPvZ& pvz) noexcept {
    if (g_restore_frame_duration > 0 && pvz.GameOn()) {
        pvz.SetFrameDuration(g_restore_frame_duration);
    }
    g_restore_frame_duration = 0;
}

void clear_headless_state() noexcept {
    g_headless_active.store(false, std::memory_order_release);
    g_restore_paused = false;
    g_restore_frame_duration = 0;
    g_headless_board = 0;
    g_headless_pid = 0;
    g_last_ui_service = {};
    g_headless_owns_main_loop.store(false, std::memory_order_release);
    request_ui_service_worker_stop();
}

void abandon_headless_nolock(
    PvzRuntime::VersionedPvZ& pvz,
    uintptr_t board) noexcept {
    const uintptr_t current_board = board_ptr_nolock(pvz);
    if (current_board && current_board == board) {
        // 战斗仍稳定时恢复进入无头模式前的暂停状态；如果已经进入终局或场景切换，
        // 必须解除暂停，让原版主线程在释放屏障后完成清理流程。
        const bool paused = pvz.GameUI() == 3 ? g_restore_paused : false;
        pvz.WriteMemory<bool>(paused, {board + pvz.data().game_paused});
    }
    restore_headless_frame_duration_nolock(pvz);
    release_headless_barrier_nolock();
    pvz.enable_hack(pvz.data().block_main_loop, false);
    clear_headless_state();
}

bool service_ui_nolock(
    PvzRuntime::VersionedPvZ& pvz,
    uintptr_t board) {
    if (!is_same_battle_nolock(pvz, board)) return false;

    // 恢复原生窗口线程前先冻结游戏逻辑。界面服务窗口内即使调用 Board::Update，
    // mPaused 也会阻止游戏推进，只允许消息、界面和渲染继续运行。
    pvz.WriteMemory<bool>(true, {board + pvz.data().game_paused});
    release_headless_barrier_nolock();

    std::this_thread::sleep_for(
        std::chrono::milliseconds(kUiServiceWindowMs));

    if (!is_same_battle_nolock(pvz, board)) return false;

    // 任何 RL 注入的 Board::Update 再次执行前，都要重新取得精确安全点。
    // 这笔同步成本约每三秒支付一次，而不是每个决策区间支付。
    if (!acquire_headless_barrier_nolock(pvz)) return false;
    if (!is_same_battle_nolock(pvz, board)) {
        release_headless_barrier_nolock();
        return false;
    }

    g_last_ui_service = std::chrono::steady_clock::now();
    g_ui_service_count.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool ui_service_due_nolock() noexcept {
    if (g_last_ui_service.time_since_epoch().count() == 0) return false;
    return std::chrono::steady_clock::now() - g_last_ui_service >=
        std::chrono::milliseconds(kUiServiceIntervalMs);
}

DWORD WINAPI ui_service_worker_main(void*) noexcept {
    g_ui_service_worker_running.store(true, std::memory_order_release);

    HANDLE stop_event = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_ui_service_worker_mutex);
        stop_event = g_ui_service_stop_event;
    }

    while (stop_event) {
        const DWORD wait = ::WaitForSingleObject(
            stop_event,
            static_cast<DWORD>(kUiServiceIntervalMs));
        if (wait != WAIT_TIMEOUT) break;

        auto& mgr = process::ProcessManager::instance();
        std::lock_guard<std::mutex> lock(mgr.mutex());
        if (!g_headless_active.load(std::memory_order_acquire)) break;
        if (!ui_service_due_nolock()) continue;

        auto& pvz = mgr.runtime();
        if (!pvz.GameOn() && !pvz.FindPvZ()) {
            g_ui_service_failures.fetch_add(1, std::memory_order_relaxed);
            release_headless_barrier_nolock();
            clear_headless_state();
            break;
        }

        if (!is_same_battle_nolock(pvz, g_headless_board)) {
            g_ui_service_failures.fetch_add(1, std::memory_order_relaxed);
            abandon_headless_nolock(pvz, g_headless_board);
            break;
        }

        if (!g_headless_owns_main_loop.load(std::memory_order_acquire) ||
            !g_headless_barrier || !g_headless_barrier->acquired() ||
            !service_ui_nolock(pvz, g_headless_board)) {
            g_ui_service_failures.fetch_add(1, std::memory_order_relaxed);
            abandon_headless_nolock(pvz, g_headless_board);
            break;
        }
    }

    g_ui_service_worker_running.store(false, std::memory_order_release);
    return 0;
}

bool start_ui_service_worker() noexcept {
    std::lock_guard<std::mutex> lock(g_ui_service_worker_mutex);
    if (g_ui_service_thread || g_ui_service_stop_event) return false;

    g_ui_service_stop_event = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!g_ui_service_stop_event) return false;

    g_ui_service_thread = ::CreateThread(
        nullptr,
        0,
        &ui_service_worker_main,
        nullptr,
        0,
        nullptr);
    if (!g_ui_service_thread) {
        ::CloseHandle(g_ui_service_stop_event);
        g_ui_service_stop_event = nullptr;
        return false;
    }
    return true;
}

bool stop_ui_service_worker() noexcept {
    HANDLE thread = nullptr;
    HANDLE stop_event = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_ui_service_worker_mutex);
        thread = g_ui_service_thread;
        stop_event = g_ui_service_stop_event;
        if (stop_event) ::SetEvent(stop_event);
    }

    if (thread) {
        if (::WaitForSingleObject(thread, 10000) != WAIT_OBJECT_0) return false;
    }

    std::lock_guard<std::mutex> lock(g_ui_service_worker_mutex);
    if (g_ui_service_thread) {
        ::CloseHandle(g_ui_service_thread);
        g_ui_service_thread = nullptr;
    }
    if (g_ui_service_stop_event) {
        ::CloseHandle(g_ui_service_stop_event);
        g_ui_service_stop_event = nullptr;
    }
    return true;
}

}  // 匿名命名空间

bool SimulationClock::begin_headless_mode() {
    HeadlessDiagnostics diag;

    // 如果上一次终局或错误路径已经让工作线程退出，先回收句柄。
    // 持有 ProcessManager 互斥锁时绝不能等待该线程。
    if (!g_headless_active.load(std::memory_order_acquire)) {
        if (!stop_ui_service_worker()) return false;
    }

    auto& mgr = process::ProcessManager::instance();
    mgr.set_background_running(true);

    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();

    if (Pt::Code::has_pending_remote_call(pvz.process_handle())) return false;

    if (g_headless_active.load(std::memory_order_acquire)) {
        diag.status = HeadlessBeginStatus::AlreadyActive;
        if (pvz.GameOn()) {
            diag.ui_before = diag.ui_after = pvz.GameUI();
            diag.board_before = diag.board_after = board_ptr_nolock(pvz);
            diag.clock_before = diag.clock_after =
                clock_for_board_nolock(pvz, diag.board_before);
            if (diag.board_before) {
                diag.paused_before = diag.paused_after = pvz.ReadMemory<bool>(
                    {diag.board_before + pvz.data().game_paused});
            }
        }
        set_diagnostics(diag);
        return true;
    }

    if (!pvz.GameOn() && !pvz.FindPvZ()) {
        diag.status = HeadlessBeginStatus::GameNotFound;
        set_diagnostics(diag);
        return false;
    }

    diag.ui_before = pvz.GameUI();
    if (diag.ui_before != 3) {
        diag.status = HeadlessBeginStatus::NotInBattle;
        diag.ui_after = diag.ui_before;
        set_diagnostics(diag);
        return false;
    }

    diag.board_before = board_ptr_nolock(pvz);
    if (!diag.board_before) {
        diag.status = HeadlessBeginStatus::BoardNull;
        diag.ui_after = pvz.GameUI();
        set_diagnostics(diag);
        return false;
    }

    if (PvzRuntime::board_update_address(pvz) == 0) {
        diag.status = HeadlessBeginStatus::UpdateFnNull;
        diag.ui_after = pvz.GameUI();
        diag.board_after = board_ptr_nolock(pvz);
        set_diagnostics(diag);
        return false;
    }

    diag.clock_before = clock_for_board_nolock(pvz, diag.board_before);
    diag.paused_before = pvz.ReadMemory<bool>(
        {diag.board_before + pvz.data().game_paused});

    // 把调度帧时长临时压到 1 毫秒，可以缩短首次屏障稳定等待，并保证短界面服务窗口内至少能渲染一帧。
    // 游戏逻辑仍由显式 Board::Update 驱动，因此 RL 决策之间不会自行推进；所有正常、终局和失败退出都会恢复原时长。
    g_restore_frame_duration = std::max(1, pvz.GetFrameDuration());
    pvz.SetFrameDuration(kHeadlessFrameDurationMs);

    // Acquire a verified safe point before writing the captured Board. Until
    // the next UI service, the remote batch is the only update executor.
    pvz.enable_hack(pvz.data().block_main_loop, false);

    if (!acquire_headless_barrier_nolock(pvz)) {
        diag.status = HeadlessBeginStatus::BarrierAcquireFailed;
        diag.ui_after = pvz.GameUI();
        diag.board_after = board_ptr_nolock(pvz);
        diag.clock_after = clock_for_board_nolock(pvz, diag.board_after);
        restore_headless_frame_duration_nolock(pvz);
        pvz.enable_hack(pvz.data().block_main_loop, false);
        set_diagnostics(diag);
        return false;
    }

    diag.ui_after = pvz.GameUI();
    diag.board_after = board_ptr_nolock(pvz);
    diag.clock_after = clock_for_board_nolock(pvz, diag.board_after);
    if (diag.board_after) {
        diag.paused_after = pvz.ReadMemory<bool>(
            {diag.board_after + pvz.data().game_paused});
    }

    if (diag.ui_after != 3 || diag.board_after != diag.board_before) {
        diag.status = HeadlessBeginStatus::BattleChanged;
        restore_headless_frame_duration_nolock(pvz);
        release_headless_barrier_nolock();
        pvz.enable_hack(pvz.data().block_main_loop, false);
        set_diagnostics(diag);
        return false;
    }

    // Only write the captured Board after the verified handoff and identity
    // check. It may have been replaced while the initial frame was in flight.
    pvz.WriteMemory<bool>(true, {diag.board_before + pvz.data().game_paused});
    g_restore_paused = diag.paused_after;
    g_headless_board = diag.board_before;
    g_headless_pid = pvz.process_id();
    g_headless_active.store(true, std::memory_order_release);
    g_last_ui_service = std::chrono::steady_clock::now();
    g_ui_service_count.store(0, std::memory_order_relaxed);
    g_ui_service_failures.store(0, std::memory_order_relaxed);

    if (!start_ui_service_worker()) {
        diag.status = HeadlessBeginStatus::UiWorkerStartFailed;
        pvz.WriteMemory<bool>(
            g_restore_paused,
            {diag.board_before + pvz.data().game_paused});
        restore_headless_frame_duration_nolock(pvz);
        release_headless_barrier_nolock();
        pvz.enable_hack(pvz.data().block_main_loop, false);
        clear_headless_state();
        set_diagnostics(diag);
        return false;
    }

    diag.status = HeadlessBeginStatus::Ok;
    set_diagnostics(diag);
    return true;
}

int SimulationClock::advance_ticks_headless(int ticks) {
    if (ticks <= 0) return 0;

    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    if (!g_headless_active.load(std::memory_order_acquire)) return -2;

    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) {
        release_headless_barrier_nolock();
        clear_headless_state();
        return -1;
    }
    if (Pt::Code::has_pending_remote_call(pvz.process_handle())) {
        abandon_headless_nolock(pvz, g_headless_board);
        return -6;
    }
    if (pvz.GameUI() != 3) {
        abandon_headless_nolock(pvz, g_headless_board);
        return 0;
    }

    const uintptr_t board = board_ptr_nolock(pvz);
    if (!board || board != g_headless_board) {
        abandon_headless_nolock(pvz, g_headless_board);
        return -3;
    }
    const uintptr_t fn = PvzRuntime::board_update_address(pvz);
    if (fn == 0) {
        abandon_headless_nolock(pvz, g_headless_board);
        return -4;
    }

    if (!g_headless_owns_main_loop.load(std::memory_order_acquire) ||
        !g_headless_barrier || !g_headless_barrier->acquired()) {
        // 只有明确拥有原生主线程时才允许注入 Board::Update。
        abandon_headless_nolock(pvz, g_headless_board);
        return -7;
    }

    // 独立工作线程负责 Python 长时间计算或暂停时的界面服务；但高密度端到端/RL 循环中，
    // observation/action API 可能持续抢到 ProcessManager 互斥锁，使工作线程长期得不到执行机会。
    // 因此在已有决策边界也检查一次是否需要界面服务。常规路径只多一次稳态时钟比较，
    // 屏障交接仍然最多每个服务周期发生一次，而不是每个 step 都发生。
    if (ui_service_due_nolock() && !service_ui_nolock(pvz, board)) {
        abandon_headless_nolock(pvz, board);
        return -7;
    }

    const std::int64_t start_clock = clock_for_board_nolock(pvz, board);
    int remaining = ticks;
    bool injection_failed = false;

    pvz.WriteMemory<bool>(false, {board + pvz.data().game_paused});

    while (remaining > 0) {
        if (!is_same_battle_nolock(pvz, board)) break;

        const int batch = std::min(remaining, kMaxInjectedBatchTicks);
        const int issued = inject_board_update_batch_nolock(pvz, board, batch, fn, 3);
        if (issued < 0) {
            injection_failed = true;
            break;
        }
        remaining -= issued;

        if (issued < batch || !is_same_battle_nolock(pvz, board)) break;
    }

    const uintptr_t current_board = board_ptr_nolock(pvz);
    if (current_board && current_board == board) {
        if (pvz.GameUI() == 3) {
            // 回到稳定决策边界时原生线程仍被挂起，Board 保持暂停，
            // 后续 observation/action 读取可以在不与原版主循环竞争的情况下执行。
            pvz.WriteMemory<bool>(true, {board + pvz.data().game_paused});
        } else {
            pvz.WriteMemory<bool>(false, {board + pvz.data().game_paused});
        }
    }

    if (injection_failed) {
        abandon_headless_nolock(pvz, board);
        return -6;
    }

    if (!is_same_battle_nolock(pvz, board)) {
        const int issued = std::max(0, ticks - remaining);
        abandon_headless_nolock(pvz, board);
        return issued;
    }

    const std::int64_t end_clock = clock_for_board_nolock(pvz, board);
    const std::int64_t delta = end_clock - start_clock;
    return delta > 0 ? static_cast<int>(delta) : 0;
}

bool SimulationClock::end_headless_mode() {
    // 取得 ProcessManager 互斥锁前先唤醒并等待看门狗结束，
    // 因为工作线程自己也可能正在等待该互斥锁完成界面服务。
    if (!stop_ui_service_worker()) return false;

    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    if (!g_headless_active.load(std::memory_order_acquire)) {
        release_headless_barrier_nolock();
        return true;
    }

    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) {
        release_headless_barrier_nolock();
        clear_headless_state();
        return false;
    }

    const uintptr_t board = board_ptr_nolock(pvz);
    if (board && board == g_headless_board && pvz.GameUI() == 3) {
        // 恢复原生线程前，先恢复进入无头模式之前要求的暂停状态。
        pvz.WriteMemory<bool>(
            g_restore_paused,
            {board + pvz.data().game_paused});
    } else if (board && board == g_headless_board) {
        pvz.WriteMemory<bool>(false, {board + pvz.data().game_paused});
    }

    restore_headless_frame_duration_nolock(pvz);
    release_headless_barrier_nolock();
    pvz.enable_hack(pvz.data().block_main_loop, false);
    clear_headless_state();
    return true;
}

void SimulationClock::release_headless_after_failure_nolock(PvzRuntime::VersionedPvZ& pvz) noexcept {
    if (g_headless_active) abandon_headless_nolock(pvz, board_ptr_nolock(pvz));
}

bool SimulationClock::can_reuse_headless_barrier_nolock(PvzRuntime::VersionedPvZ& pvz) noexcept {
    return g_headless_active.load(std::memory_order_acquire) &&
        g_headless_owns_main_loop.load(std::memory_order_acquire) &&
        g_headless_pid && pvz.process_id() == g_headless_pid &&
        pvz.process_handle() && GetProcessId(pvz.process_handle()) == g_headless_pid &&
        g_headless_barrier && g_headless_barrier->owns_process(g_headless_pid) &&
        pvz.GameOn() && pvz.GameUI() == 3 && g_headless_board &&
        board_ptr_nolock(pvz) == g_headless_board;
}

bool SimulationClock::headless_owns_main_loop() noexcept {
    return g_headless_owns_main_loop.load(std::memory_order_acquire);
}

int SimulationClock::headless_begin_status() {
    std::lock_guard<std::mutex> lock(g_diag_mutex);
    return static_cast<int>(g_diag.status);
}

std::string SimulationClock::headless_diagnostics_json() {
    HeadlessDiagnostics diag;
    {
        std::lock_guard<std::mutex> lock(g_diag_mutex);
        diag = g_diag;
    }

    std::ostringstream ss;
    ss << "{";
    ss << "\"status\":" << static_cast<int>(diag.status) << ",";
    ss << "\"status_name\":\"" << begin_status_name(diag.status) << "\",";
    ss << "\"ui_before\":" << diag.ui_before << ",";
    ss << "\"ui_after\":" << diag.ui_after << ",";
    ss << "\"board_before\":\"0x" << std::hex << diag.board_before << std::dec << "\",";
    ss << "\"board_after\":\"0x" << std::hex << diag.board_after << std::dec << "\",";
    ss << "\"clock_before\":" << diag.clock_before << ",";
    ss << "\"clock_after\":" << diag.clock_after << ",";
    ss << "\"paused_before\":" << (diag.paused_before ? "true" : "false") << ",";
    ss << "\"paused_after\":" << (diag.paused_after ? "true" : "false") << ",";
    ss << "\"owns_main_loop\":"
       << (g_headless_owns_main_loop.load(std::memory_order_acquire) ? "true" : "false") << ",";
    ss << "\"headless_active\":"
       << (g_headless_active.load(std::memory_order_acquire) ? "true" : "false") << ",";
    ss << "\"ui_service_worker_running\":"
       << (g_ui_service_worker_running.load(std::memory_order_acquire) ? "true" : "false") << ",";
    ss << "\"ui_service_count\":"
       << g_ui_service_count.load(std::memory_order_relaxed) << ",";
    ss << "\"ui_service_failures\":"
       << g_ui_service_failures.load(std::memory_order_relaxed) << ",";
    ss << "\"barrier_last_instruction\":" << process::MainLoopBarrier::last_instruction() << ",";
    ss << "\"barrier_retry_count\":" << process::MainLoopBarrier::retry_count() << ",";
    ss << "\"barrier_failure_count\":" << process::MainLoopBarrier::failure_count() << ",";
    ss << "\"ui_service_interval_ms\":" << kUiServiceIntervalMs << ",";
    ss << "\"ui_service_window_ms\":" << kUiServiceWindowMs << ",";
    ss << "\"headless_frame_duration_ms\":" << kHeadlessFrameDurationMs;
    ss << "}";
    return ss.str();
}

}  // 命名空间 pvz::game
