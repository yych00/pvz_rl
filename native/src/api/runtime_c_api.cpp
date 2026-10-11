#include "pvz/api.h"
#include "pvz/runtime.h"
#include "serializer.h"
#include "transition_result_buffer.h"
#include "../runtime/turbo_runtime.h"
#include "../runtime/native_runtime.h"
#include "../runtime/audio_mute.h"
#include "wire_result.h"
#include "binary_serializer.h"
#include "step_timing.h"
#include "runtime_error.h"
#include "../runtime/resident_executor.h"
#include "runtime/native_errors.h"

namespace {

constexpr char kCapabilities[] =
    R"({"interface_version":1,"modes":["stepped","headless","continuous","turbo"],)"
    R"("features":["result_requests","step_profiling","batch_plant_read","attach_pid",)"
    R"("resident_executor","binary_transport","shared_memory_transport","shared_snapshot","phase_advance",)"
    R"("observation_quality","boss_projectiles"],"observation_schema_version":4})";

void record_error(const char* operation, const char* kind, int code,
                  const char* message) noexcept {
    // Diagnostics are secondary: allocation/read failure cannot replace the error.
    try {
        pvz::api::record_runtime_error(operation, kind, code, message,
            pvz::runtime::ResidentExecutor::instance().diagnostics_json());
    } catch (...) {}
}

}  // namespace

extern "C" {

int PVZ_GYM_CALL pvz_gym_get_runtime_error_json(char* out, int capacity) {
    try { return pvz::api::Serializer::copy_to_c_buffer(pvz::api::runtime_error_json, out, capacity); }
    catch (...) { return -1; }
}

int PVZ_GYM_CALL pvz_gym_get_runtime_capabilities_json(char* out, int capacity) {
    try {
        return pvz::api::Serializer::copy_to_c_buffer(kCapabilities, out, capacity);
    } catch (...) { return -1; }
}

int PVZ_GYM_CALL pvz_gym_configure_runtime(
    std::uint32_t process_id, int executor, int transport, int snapshot) {
    using pvz::runtime::Status;
    if (executor < 0 || executor > 1 || transport < 0 || transport > 2 || snapshot < 0 || snapshot > 1)
        return static_cast<int>(Status::InvalidArgument);
    pvz::runtime::RuntimeOptions options{process_id, static_cast<pvz::runtime::ExecutorKind>(executor),
        static_cast<pvz::runtime::TransportKind>(transport), static_cast<pvz::runtime::SnapshotKind>(snapshot)};
    try { return static_cast<int>(pvz::runtime::NativeRuntime::instance().configure(options)); }
    catch (...) { return static_cast<int>(Status::InvalidState); }
}

int PVZ_GYM_CALL pvz_gym_begin_turbo_mode(std::uint32_t flags) {
    using namespace pvz::runtime;
    pvz::api::clear_runtime_error();
    if (flags & ~kTurboFlags) return static_cast<int>(Status::InvalidArgument);
    auto& runtime = TurboRuntime::instance();
    const auto configured = runtime.set_options({flags});
    if (configured != Status::Ok) return static_cast<int>(configured);
    try {
        const auto status = runtime.enter(NativeRuntime::instance());
        if (status != Status::Ok) record_error("begin_turbo_mode", "status",
            static_cast<int>(status), runtime.enter_error().c_str());
        return static_cast<int>(status);
    } catch (const std::exception& error) {
        record_error("begin_turbo_mode", "exception", -1, error.what());
    } catch (...) { record_error("begin_turbo_mode", "exception", -1, "unknown C++ exception"); }
    // Freeze diagnostics before cleanup changes resident ownership/mailbox state.
    try { runtime.leave(); } catch (...) {}
    return static_cast<int>(Status::InvalidState);
}

int PVZ_GYM_CALL pvz_gym_end_turbo_mode() {
    try { return static_cast<int>(pvz::runtime::TurboRuntime::instance().leave()); }
    catch (...) { return static_cast<int>(pvz::runtime::Status::InvalidState); }
}
int PVZ_GYM_CALL pvz_gym_close_runtime() {
    try { return static_cast<int>(pvz::runtime::NativeRuntime::instance().close()); }
    catch (...) { return static_cast<int>(pvz::runtime::Status::InvalidState); }
}

int PVZ_GYM_CALL pvz_gym_set_process_audio_muted(std::uint32_t process_id, bool mute) {
    try {
        if (mute) {
            pvz::runtime::AudioMute::instance().mute_process(process_id);
        } else {
            pvz::runtime::AudioMute::instance().unmute_process(process_id);
        }
        return 0;
    } catch (...) {
        return -1;
    }
}

int PVZ_GYM_CALL pvz_gym_env_step_turbo(
    int action_type, int card_slot, int row, int col,
    int shovel_target, int target_row, int target_col,
    int ticks, char* out, int capacity) {
    pvz::api::StepTiming timing;
    pvz::api::StepTimingScope scope(timing);
    pvz::api::clear_runtime_error();
    try {
        pvz::StepResult result;
        pvz::Action action;
        action.type = static_cast<pvz::ActionType>(action_type);
        action.card_slot = card_slot;
        action.row = row;
        action.col = col;
        action.shovel_target = static_cast<pvz::ShovelTarget>(shovel_target);
        action.target_row = target_row;
        action.target_col = target_col;
        pvz::runtime::TurboRuntime::instance().step(action, ticks, result);
        // Same one-execution/result-read contract as the existing step APIs.
        return timing.measure(pvz::api::StepTiming::Serialize, [&] {
            return pvz::api::transition_results.publish(pvz::api::wire_result(result), out, capacity);
        });
    } catch (const PvzRuntime::NativeOperationError& error) {
        record_error("step_turbo", "native_operation", static_cast<int>(error.error()), error.what());
    } catch (const std::exception& error) {
        record_error("step_turbo", "exception", -1, error.what());
    } catch (...) { record_error("step_turbo", "exception", -1, "unknown C++ exception"); }
    return -1;
}

int PVZ_GYM_CALL pvz_gym_advance_phase(int phase, int updates, char* out, int capacity) {
    try {
        pvz::runtime::PhaseAdvanceResult result;
        const auto status = pvz::runtime::NativeRuntime::instance().advance_phase(
            {static_cast<pvz::game::RuntimePhase>(phase), updates}, result);
        const auto payload = "{\"status\":" + std::to_string(static_cast<int>(status)) +
            ",\"phase_before\":" + std::to_string(static_cast<int>(result.phase_before)) +
            ",\"phase_after\":" + std::to_string(static_cast<int>(result.phase_after)) +
            ",\"updates_executed\":" + std::to_string(result.updates_executed) +
            ",\"game_ticks_advanced\":" + std::to_string(result.game_ticks_advanced) + "}";
        return pvz::api::transition_results.publish(payload, out, capacity);
    } catch (...) { return -1; }
}

int PVZ_GYM_CALL pvz_gym_read_runtime_snapshot(char* out, int capacity) {
    try {
        pvz::runtime::RuntimeSnapshot snapshot;
        if (pvz::runtime::NativeRuntime::instance().read_snapshot(snapshot) != pvz::runtime::Status::Ok) return -1;
        pvz::api::BinaryWriter writer;
        const auto at = writer.begin_map(); std::uint32_t count = 0;
        writer.field("process_id", snapshot.identity.process_id, count);
        writer.field("episode_id", snapshot.identity.episode_id, count);
        writer.field("board_id", snapshot.identity.board_id, count);
        writer.field("revision", snapshot.identity.revision, count);
        writer.field("observation", snapshot.observation, count);
        writer.field("action_mask", snapshot.action_mask, count); writer.end_map(at, count);
        return pvz::api::Serializer::copy_to_c_buffer(writer.bytes, out, capacity);
    } catch (...) { return -1; }
}

}  // extern "C"
