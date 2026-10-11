#include "main_loop_barrier.h"
#include "thread_safe_point.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>

namespace pvz::process {
namespace {

std::atomic<std::uint32_t> g_last_instruction{0};
std::atomic<std::uint64_t> g_retry_count{0};
std::atomic<std::uint64_t> g_failure_count{0};

bool is_short_branch_opcode(std::uint8_t opcode) noexcept {
    // 当前支持的 PvZ 版本在 block_main_loop 使用 8 位短条件跳转或短跳转；
    // 把位移字节改成 0xFE 后，跳转目标会指回自身形成停车点。
    return (opcode >= 0x70 && opcode <= 0x7f) || opcode == 0xeb;
}

}  // 匿名命名空间

MainLoopBarrier::MainLoopBarrier(PvzRuntime::VersionedPvZ& pvz) noexcept
    : pvz_(pvz) {}

MainLoopBarrier::~MainLoopBarrier() {
    release();
}

bool MainLoopBarrier::acquire() noexcept {
    if (acquired_) return true;
    if (!pvz_.GameOn()) return false;

    const auto patch = pvz_.data().block_main_loop;
    if (patch.mem_addr <= 1) return false;

    HWND window = pvz_.game_window();
    if (!window || !::IsWindow(window)) return false;

    // 若游戏窗口被最小化，主循环可能处于完全休眠或消息泵停滞状态；
    // 必须在施加自跳转补丁与挂起前静默恢复窗口，使主线程正常运转越过补丁点。
    if (::IsIconic(window)) {
        ::ShowWindowAsync(window, SW_SHOWNOACTIVATE);
        ::Sleep(40);
    }

    DWORD process_id = 0;
    const DWORD thread_id = ::GetWindowThreadProcessId(window, &process_id);
    if (!thread_id || !process_id) return false;

    const DWORD expected_process_id = ::GetProcessId(pvz_.process_handle());
    if (expected_process_id && process_id != expected_process_id) return false;

    thread_ = ::OpenThread(
        THREAD_SUSPEND_RESUME | THREAD_QUERY_INFORMATION | THREAD_GET_CONTEXT | THREAD_SET_CONTEXT | SYNCHRONIZE,
        FALSE,
        thread_id);
    if (!thread_) return false;

    // The patch must park the thread outside update/draw/lifecycle callbacks.
    // Its presence alone does not prove the thread has reached that point.
    pvz_.enable_hack(patch, true);
    patch_enabled_ = true;

    const std::uintptr_t spin_ip = patch.mem_addr - 1;
    const std::uint8_t opcode = pvz_.ReadMemory<std::uint8_t>({spin_ip});
    const std::uint8_t displacement =
        pvz_.ReadMemory<std::uint8_t>({patch.mem_addr});
    if (!is_short_branch_opcode(opcode) || displacement != 0xfe) {
        release();
        return false;
    }

    // 稳健交接主线程所有权：先启用阻塞补丁，再等待已经越过补丁的当前帧结束，
    // 最后挂起已经停在补丁处的窗口线程。同步成本只在无头模式启动或周期性界面服务时支付，
    // 而不是每个 RL 决策都支付。即使帧时长被压到 1 毫秒，也给在途 Board::Update 留出足够时间。
    const int frame_ms = std::max(1, pvz_.GetFrameDuration());
    const int settle_ms = std::max(20, frame_ms * 3);
    std::this_thread::sleep_for(std::chrono::milliseconds(settle_ms));

    // Normally one context read per handoff suffices. Only an in-flight frame
    // needs retries; resume it so it can finish instead of injecting into it.
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(500);
    do {
        if (::WaitForSingleObject(thread_, 0) == WAIT_OBJECT_0) break;
        DWORD observed_ip = 0;
        const auto status = try_suspend_at_safe_point(thread_, spin_ip, &observed_ip);
        g_last_instruction.store(observed_ip, std::memory_order_relaxed);
        if (status == SafePointStatus::Failed) break;
        if (status == SafePointStatus::Acquired) {
            acquired_ = true;
            if (pvz_.ReadMemory<std::uint8_t>({spin_ip}) == opcode &&
                pvz_.ReadMemory<std::uint8_t>({patch.mem_addr}) == 0xfe)
                return true;
            break;
        }
        g_retry_count.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);
    release();
    g_failure_count.fetch_add(1, std::memory_order_relaxed);
    return false;
}

std::uint32_t MainLoopBarrier::last_instruction() noexcept {
    return g_last_instruction.load(std::memory_order_relaxed);
}

std::uint64_t MainLoopBarrier::retry_count() noexcept {
    return g_retry_count.load(std::memory_order_relaxed);
}

std::uint64_t MainLoopBarrier::failure_count() noexcept {
    return g_failure_count.load(std::memory_order_relaxed);
}

void MainLoopBarrier::release() noexcept {
    // 先恢复原始机器码，再恢复可能停在补丁跳转处的线程，保证下一条指令看到原版代码。
    if (patch_enabled_) {
        pvz_.enable_hack(pvz_.data().block_main_loop, false);
        patch_enabled_ = false;
    }

    if (acquired_ && thread_) {
        ::ResumeThread(thread_);
    }
    acquired_ = false;

    if (thread_) {
        ::CloseHandle(thread_);
        thread_ = nullptr;
    }
}

bool MainLoopBarrier::handoff(std::uint32_t entry) noexcept {
    if (!acquired_ || !entry || !thread_) return false;
#if defined(_WIN64)
    WOW64_CONTEXT context{};
    context.ContextFlags = WOW64_CONTEXT_CONTROL;
    if (!::Wow64GetThreadContext(thread_, &context)) return false;
    context.Eip = entry;
    if (!::Wow64SetThreadContext(thread_, &context)) return false;
#else
    CONTEXT context{};
    context.ContextFlags = CONTEXT_CONTROL;
    if (!::GetThreadContext(thread_, &context)) return false;
    context.Eip = entry;
    if (!::SetThreadContext(thread_, &context)) return false;
#endif
    release();
    return true;
}

}  // 命名空间 pvz::process
