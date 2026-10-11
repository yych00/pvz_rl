#include "turbo_runtime.h"
#include "resident_executor.h"
#include "../game/simulation_clock.h"
#include "../process/process_manager.h"
#include "audio_mute.h"
#include "runtime/pvz_runtime_layout.h"

namespace pvz::runtime {

TurboRuntime& TurboRuntime::instance() { static TurboRuntime runtime; return runtime; }
Status TurboRuntime::enter(ExecutionBackend& backend) {
    enter_error_.clear();
    if (options_.flags & ~kTurboFlags) return Status::InvalidArgument;
    if (!(options_.flags & SuppressRender) || !(options_.flags & ServiceMessages)) return Status::InvalidArgument;
    if (active()) return Status::Ok;
    if (!game::SimulationClock::end_headless_mode()) {
        enter_error_ = "headless clock release failed"; return Status::InvalidState;
    }
    auto& manager = process::ProcessManager::instance();
    if (!manager.attach()) {
        enter_error_ = "target game process is unavailable"; return Status::InvalidState;
    }
    std::lock_guard<std::mutex> lock(manager.mutex());
    auto& pvz = manager.runtime();
    if (PvzRuntime::board_update_address(pvz) == 0 || pvz.version_id() != PVZ_1_2_0_1065_EN)
        return Status::Unsupported;
    auto& executor = ResidentExecutor::instance();
    if (executor.owns(pvz) && !executor.stop(pvz)) {
        enter_error_ = "previous resident executor could not stop; ownership remains quarantined";
        return Status::InvalidState;
    }
    original_board_ = pvz.ReadMemory<std::uint32_t>({pvz.data().lawn, pvz.data().board});
    original_pause_ = original_board_ && pvz.ReadMemory<bool>({original_board_ + pvz.data().game_paused});
    if (!executor.begin(pvz, true)) {
        enter_error_ = "resident executor could not acquire or initialize the game main thread";
        return Status::InvalidState;
    }
    if (options_.flags & MuteAudio) AudioMute::instance().begin(pvz.process_id());
    backend_ = &backend;
    return Status::Ok;
}

Status TurboRuntime::leave() {
    if (!active()) return Status::Ok;
    auto& manager = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(manager.mutex());
    auto& pvz = manager.runtime();
    const auto board = pvz.ReadMemory<std::uint32_t>({pvz.data().lawn, pvz.data().board});
    if (board) pvz.WriteMemory<bool>(board == original_board_ && original_pause_, {board + pvz.data().game_paused});
    const bool stopped = ResidentExecutor::instance().stop(pvz);
    AudioMute::instance().end(); backend_ = nullptr;
    return stopped ? Status::Ok : Status::InvalidState;
}
bool TurboRuntime::service() {
    if (!active()) return true;
    auto& manager = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(manager.mutex());
    return ResidentExecutor::instance().service();
}

Status TurboRuntime::advance_phase(const PhaseAdvanceRequest& request, PhaseAdvanceResult& result) {
    result = {};
    return backend_ ? backend_->advance_phase(request, result) : Status::InvalidState;
}

Status TurboRuntime::step(const Action& action, int ticks, StepResult& result) {
    result = {};
    if (backend_) return backend_->step(action, ticks, result);
    result.action = {false, ActionError::GameNotInBattleState, "Turbo is not active"};
    result.error = ticks > 0 ? ActionError::GameNotInBattleState : ActionError::InvalidArgument;
    result.message = "begin Turbo before executing a transition";
    return ticks > 0 ? Status::InvalidState : Status::InvalidArgument;
}

}  // namespace pvz::runtime
