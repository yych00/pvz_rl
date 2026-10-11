#pragma once

#include "runtime/pvz_runtime_layout.h"
#include "barrier_ownership.h"

#include <Windows.h>

namespace pvz::process {

// 原版 PvZ 主线程同步屏障。
//
// 工具箱的 block_main_loop 补丁会把短跳转位移改成 0xFE，使原生主循环停在自跳转处。
// acquire() 先启用补丁，等待已经进入当前帧的逻辑自然结束，再挂起窗口线程。
// 挂起后核对 x86 EIP，只有确实停在屏障才交出主线程所有权；
// 尚未到达时恢复线程并有限重试，失败时禁止远程更新。
//
// 普通调用只应短暂持有屏障完成远程操作。高速无头训练会跨多个决策区间持续持有，
// 只在周期性界面/消息泵服务时释放；释放长期持有者前必须保证 Board::mPaused 为 true。
class MainLoopBarrier {
public:
    explicit MainLoopBarrier(PvzRuntime::VersionedPvZ& pvz) noexcept;
    ~MainLoopBarrier();

    MainLoopBarrier(const MainLoopBarrier&) = delete;
    MainLoopBarrier& operator=(const MainLoopBarrier&) = delete;

    bool acquire() noexcept;
    void release() noexcept;
    // Redirect the parked x86 owner into a resident message/command loop.
    bool handoff(std::uint32_t entry) noexcept;

    bool acquired() const noexcept { return acquired_; }
    bool owns_process(DWORD pid) const noexcept {
        return owns_live_thread(acquired_, thread_, pid);
    }
    static std::uint32_t last_instruction() noexcept;
    static std::uint64_t retry_count() noexcept;
    static std::uint64_t failure_count() noexcept;

private:
    PvzRuntime::VersionedPvZ& pvz_;
    HANDLE thread_ = nullptr;
    bool patch_enabled_ = false;
    bool acquired_ = false;
};

}  // 命名空间 pvz::process
