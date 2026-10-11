#include "runtime/pvz_runtime_layout.h"
#include "../game/simulation_clock.h"
#include "../runtime/resident_executor.h"
#include "main_loop_barrier.h"
#include <memory>
#include <cstring>
#include <stdexcept>

void PvzRuntime::VersionedPvZ::ensure_remote_idle() {
    if (Pt::Code::has_pending_remote_call(process_handle())) {
        throw NativeOperationError(pvz::ActionError::ActionTimeout,
            "previous native injection is still running; refusing another operation");
    }
}

void PvzRuntime::VersionedPvZ::asm_code_inject() {
    asm_code_inject(false);
}

void PvzRuntime::VersionedPvZ::asm_code_inject(bool main_loop_already_frozen) {
    ensure_remote_idle();
    auto& executor = pvz::runtime::ResidentExecutor::instance();
    if (executor.owns(*this)) {
        std::unique_ptr<pvz::process::MainLoopBarrier> barrier;
        if (!main_loop_already_frozen && !executor.owner() && !pvz::game::SimulationClock::can_reuse_headless_barrier_nolock(*this)) {
            barrier = std::make_unique<pvz::process::MainLoopBarrier>(*this);
            if (!barrier->acquire()) {
                throw NativeOperationError(pvz::ActionError::ActionTimeout,
                    "resident injection could not acquire game thread");
            }
        }
        std::uint32_t result = 0;
        if (!executor.execute(code, length, calls_pos, result)) {
            throw NativeOperationError(pvz::ActionError::ActionTimeout,
                "resident execution failed; transition was not retried");
        }
        return;
    }

    std::unique_ptr<pvz::process::MainLoopBarrier> barrier;
    if (!main_loop_already_frozen &&
        !pvz::game::SimulationClock::can_reuse_headless_barrier_nolock(*this)) {
        barrier = std::make_unique<pvz::process::MainLoopBarrier>(*this);
        if (!barrier->acquire()) {
            throw NativeOperationError(pvz::ActionError::ActionTimeout,
                "native injection could not acquire the main-loop barrier");
        }
    }
    DWORD exit_code = 0;
    if (!Pt::Code::asm_code_inject(process_handle(), 5000, &exit_code)) {
        const DWORD error = ::GetLastError();
        pvz::game::SimulationClock::release_headless_after_failure_nolock(*this);
        throw NativeOperationError(
            error == ERROR_TIMEOUT || error == ERROR_BUSY
                ? pvz::ActionError::ActionTimeout : pvz::ActionError::Unsupported,
            "native injection failed (Win32 error " + std::to_string(error) + ")");
    }
    // Windows reports unhandled remote exceptions as fatal NTSTATUS exit codes.
    if (exit_code >= 0xC0000000UL) {
        std::uint32_t last_call = 0;
        if (!calls_pos.empty() && calls_pos.back() + sizeof(last_call) <= length)
            std::memcpy(&last_call, code + calls_pos.back(), sizeof(last_call));
        pvz::game::SimulationClock::release_headless_after_failure_nolock(*this);
        throw NativeOperationError(pvz::ActionError::Unsupported,
            "native injection thread failed (exit code " + std::to_string(exit_code) +
            ", last call " + std::to_string(last_call) + ")");
    }
}
