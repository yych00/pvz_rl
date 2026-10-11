#include "native_runtime.h"
#include "resident_executor.h"
#include "turbo_runtime.h"
#include "../api/shared_env.h"
#include "../process/process_manager.h"
#include "../game/runtime_phase.h"
#include "../game/simulation_clock.h"
#include "../observation/observation_builder.h"
#include "../automation/automation_controller.h"
#include "../process/main_loop_barrier.h"

namespace pvz::runtime {
NativeRuntime& NativeRuntime::instance() { static NativeRuntime runtime; return runtime; }
RuntimeCapabilities NativeRuntime::capabilities() const noexcept {
    RuntimeCapabilities caps;
    caps.stepped = caps.headless = caps.continuous = caps.attach_pid = true;
    caps.resident_executor = caps.shared_snapshot = caps.turbo = true;
    caps.binary_transport = caps.shared_memory_transport = true;
    return caps;
}
Status NativeRuntime::configure(const RuntimeOptions& options) {
    if (static_cast<int>(options.executor) < 0 || static_cast<int>(options.executor) > 1 ||
        static_cast<int>(options.transport) < 0 || static_cast<int>(options.transport) > 2 ||
        static_cast<int>(options.snapshot) < 0 || static_cast<int>(options.snapshot) > 1) return Status::InvalidArgument;
    auto& manager = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(manager.mutex());
    auto& pvz = manager.runtime();
    if (ResidentExecutor::instance().owns(pvz) && (options.process_id != options_.process_id ||
        options.executor != options_.executor)) return Status::InvalidState;
    pvz.BindProcess(options.process_id);
    options_ = options;
    new_episode();
    return Status::Ok;
}
bool NativeRuntime::prepare() {
    if (options_.executor != ExecutorKind::Resident) return true;
    auto& manager = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(manager.mutex());
    auto& pvz = manager.runtime();
    if (!pvz.GameOn()) return false;
    auto& executor = ResidentExecutor::instance();
    return executor.owns(pvz) || executor.begin(pvz, false);
}
bool NativeRuntime::begin_lifecycle() {
    lifecycle_error_.clear();
    // Native reset must release a persistent clock even when Python's cached
    // active flag is stale, or the caller uses the legacy executor directly.
    if (!game::SimulationClock::end_headless_mode()) {
        lifecycle_error_ = "headless clock release failed";
        return false;
    }
    if (TurboRuntime::instance().active() || options_.executor != ExecutorKind::Resident) return true;
    auto& manager = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(manager.mutex());
    auto& pvz = manager.runtime();
    auto& executor = ResidentExecutor::instance();
    if (executor.owner() && executor.owns(pvz)) {
        if (executor.hold()) return true;
        lifecycle_error_ = "resident owner hold failed";
        return false;
    }
    if (!executor.stop(pvz)) {
        lifecycle_error_ = "resident RPC stop failed; ownership remains quarantined";
        return false;
    }
    const auto failures = process::MainLoopBarrier::failure_count();
    if (executor.begin(pvz, true)) return true;
    lifecycle_error_ = "resident main-loop acquire failed; barrier_last_instruction=" +
        std::to_string(process::MainLoopBarrier::last_instruction()) +
        "; barrier_failure_delta=" + std::to_string(process::MainLoopBarrier::failure_count()-failures);
    return false;
}
bool NativeRuntime::finish_lifecycle() {
    if (TurboRuntime::instance().active() || !ResidentExecutor::instance().owner()) return true;
    auto& manager = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(manager.mutex());
    auto& pvz = manager.runtime();
    return ResidentExecutor::instance().stop(pvz) && ResidentExecutor::instance().begin(pvz, false);
}
bool NativeRuntime::service() {
    auto& manager = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(manager.mutex());
    return !ResidentExecutor::instance().owner() || ResidentExecutor::instance().service();
}
void NativeRuntime::new_episode() noexcept { ++episode_; invalidate(); }
Status NativeRuntime::close() {
    if (TurboRuntime::instance().leave() != Status::Ok) return Status::InvalidState;
    if (!game::SimulationClock::end_headless_mode()) return Status::InvalidState;
    auto& manager = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(manager.mutex());
    invalidate();
    return ResidentExecutor::instance().stop(manager.runtime()) ? Status::Ok : Status::InvalidState;
}
void NativeRuntime::cache_transition(const Observation& observation, const std::vector<bool>& mask) {
    invalidate();
    if (options_.snapshot != SnapshotKind::Shared || !ResidentExecutor::instance().owner() ||
        observation.quality != "fresh") return;
    auto& manager = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(manager.mutex());
    auto& pvz = manager.runtime();
    const auto state = game::RuntimePhaseReader::read_nolock(pvz);
    if (state.board_address != observation.global.board_id ||
        state.game_clock != observation.global.game_clock || state.status != observation.global.status) return;
    snapshot_ = {{pvz.process_id(), episode_, observation.global.board_id, revision_}, observation, mask};
    snapshot_commands_ = ResidentExecutor::instance().commands(); cached_ = true;
}
Status NativeRuntime::reset(const ResetConfig& config, ResetResult& result) {
    result = api::shared_env().reset(config);
    return result.success ? Status::Ok : Status::InvalidState;
}
Status NativeRuntime::step(const Action& action, int ticks, StepResult& result) {
    if (ticks <= 0) return Status::InvalidArgument;
    result = api::shared_env().step(action, ticks);
    return result.success ? Status::Ok : Status::InvalidState;
}
Status NativeRuntime::advance_phase(const PhaseAdvanceRequest& request, PhaseAdvanceResult& result) {
    if (request.updates <= 0) return Status::InvalidArgument;
    const auto before = game::RuntimePhaseReader::read();
    result.phase_before = before.phase;
    if (request.expected_phase != game::RuntimePhase::Unknown && request.expected_phase != before.phase)
        return Status::InvalidState;
    if (before.raw_scene == 2 && before.phase != game::RuntimePhase::SeedSelection) {
        result.updates_executed = game::SimulationClock::advance_intro_ticks(request.updates);
    } else if (before.phase == game::RuntimePhase::Battle) {
        result.updates_executed = game::SimulationClock::advance_ticks(request.updates);
        result.game_ticks_advanced = result.updates_executed;
    } else if (before.raw_scene == 4) {
        result.updates_executed = game::SimulationClock::advance_terminal_updates(request.updates);
    } else return Status::InvalidState;
    result.phase_after = game::RuntimePhaseReader::read().phase; invalidate();
    return result.updates_executed >= 0 ? Status::Ok : Status::InvalidState;
}
Status NativeRuntime::read_snapshot(RuntimeSnapshot& snapshot) {
    if (read_cached_snapshot(snapshot)) return Status::Ok;
    if (!observation::ObservationBuilder::build(snapshot.observation)) return Status::InvalidState;
    snapshot.action_mask = api::shared_env().get_action_mask();
    cache_transition(snapshot.observation, snapshot.action_mask);
    auto& manager = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(manager.mutex());
    auto& pvz = manager.runtime();
    snapshot.identity = {pvz.process_id(), episode_, snapshot.observation.global.board_id, revision_};
    return Status::Ok;
}
bool NativeRuntime::read_cached_snapshot(RuntimeSnapshot& snapshot) {
    if (!cached_ || !ResidentExecutor::instance().owner()) return false;
    {
        auto& manager = process::ProcessManager::instance();
        std::lock_guard<std::mutex> lock(manager.mutex());
        auto& pvz = manager.runtime();
        const auto state = game::RuntimePhaseReader::read_nolock(pvz);
        const auto board = state.board_address;
        if (cached_ && ResidentExecutor::instance().owner() && snapshot_.identity.process_id == pvz.process_id() &&
            snapshot_.identity.episode_id == episode_ && snapshot_.identity.board_id == board &&
            snapshot_.identity.revision == revision_ && snapshot_commands_ == ResidentExecutor::instance().commands() &&
            state.status == GameStatus::Playing &&
            board && state.game_clock == snapshot_.observation.global.game_clock &&
            pvz.read_memory_checked<bool>({board + pvz.data().game_paused}) == snapshot_.observation.global.is_paused &&
            pvz.read_memory_checked<int>({board + pvz.data().sun}) == snapshot_.observation.global.sun) {
            snapshot = snapshot_; return true;
        }
    }
    return false;
}
} // namespace pvz::runtime
