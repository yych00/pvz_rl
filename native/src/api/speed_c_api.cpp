#include "runtime/native_errors.h"
#include "pvz/api.h"
#include "pvz/types.h"

#include "serializer.h"
#include "transition_result_buffer.h"
#include "wire_result.h"
#include "step_timing.h"
#include "../action/action_codec.h"
#include "../action/action_executor.h"
#include "../action/action_mask.h"
#include "../automation/automation_controller.h"
#include "../env/events_tracker.h"
#include "../env/transition_observation.h"
#include "../game/runtime_phase.h"
#include "../game/simulation_clock.h"
#include "../observation/observation_builder.h"
#include "../process/process_manager.h"

#include <algorithm>
#include <mutex>
#include <string>
#include <vector>

namespace {

pvz::Action make_action(int action_type, int card_slot, int row, int col,
                        int shovel_target, int target_row, int target_col) {
    pvz::Action action;
    action.type = static_cast<pvz::ActionType>(action_type);
    action.card_slot = card_slot;
    action.row = row;
    action.col = col;
    action.shovel_target = static_cast<pvz::ShovelTarget>(shovel_target);
    action.target_row = target_row;
    action.target_col = target_col;
    return action;
}

}  // namespace

extern "C" {

int PVZ_GYM_CALL pvz_gym_get_frame_duration() {
    auto& mgr = pvz::process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return -1;
    return pvz.GetFrameDuration();
}

bool PVZ_GYM_CALL pvz_gym_set_frame_duration(int time_ms) {
    if (time_ms <= 0) return false;
    auto& mgr = pvz::process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return false;
    pvz.SetFrameDuration(time_ms);
    return pvz.GetFrameDuration() == time_ms;
}

bool PVZ_GYM_CALL pvz_gym_begin_headless_mode() {
    return pvz::game::SimulationClock::begin_headless_mode();
}

int PVZ_GYM_CALL pvz_gym_advance_ticks_headless(int ticks) {
    return pvz::game::SimulationClock::advance_ticks_headless(ticks);
}

bool PVZ_GYM_CALL pvz_gym_end_headless_mode() {
    return pvz::game::SimulationClock::end_headless_mode();
}

int PVZ_GYM_CALL pvz_gym_get_headless_begin_status() {
    return pvz::game::SimulationClock::headless_begin_status();
}

int PVZ_GYM_CALL pvz_gym_get_headless_diagnostics_json(
    char* out_json_buf, int max_len) {
    try {
        return pvz::api::Serializer::copy_to_c_buffer(
            pvz::game::SimulationClock::headless_diagnostics_json(),
            out_json_buf,
            max_len);
    } catch (...) {
        return -1;
    }
}

void PVZ_GYM_CALL pvz_gym_set_step_profiling(bool enabled) {
    pvz::api::step_profiling.store(enabled, std::memory_order_relaxed);
}
int PVZ_GYM_CALL pvz_gym_get_step_timing_json(char* out, int capacity) {
    return pvz::api::Serializer::copy_to_c_buffer(pvz::api::last_step_timing, out, capacity);
}

int PVZ_GYM_CALL pvz_gym_env_step_headless(
    int action_type, int card_slot, int row, int col,
    int shovel_target, int target_row, int target_col,
    int ticks, char* out_result_json, int max_len) {
    pvz::api::StepTiming timing;
    using T = pvz::api::StepTiming;
    auto publish = [&](const pvz::StepResult& res, char* out, int capacity) {
        return timing.measure(T::Serialize, [&] {
            return pvz::api::transition_results.publish(pvz::api::wire_result(res), out, capacity);
        });
    };
    try {
        pvz::StepResult result;
    result.action = {false, pvz::ActionError::Unsupported, "transition not executed"};

    if (ticks <= 0) {
        result.error = pvz::ActionError::InvalidArgument;
        result.message = "ticks must be positive";
        return publish(result, out_result_json, max_len);
    }

    const auto initial_state = pvz::game::RuntimePhaseReader::read();
    if (initial_state.status != pvz::GameStatus::Playing) {
        result.error = pvz::ActionError::GameNotInBattleState;
        result.message = "headless step requires an active battle";
        result.status = initial_state.status;
        return publish(result, out_result_json, max_len);
    }

    pvz::Observation before;
    if (!timing.measure(T::Before, [&] { return pvz::observation::ObservationBuilder::build(before); })) {
        result.error = pvz::ActionError::Unsupported;
        result.message = "pre-step observation is unavailable";
        return publish(result, out_result_json, max_len);
    }

    const pvz::Action action = make_action(
        action_type, card_slot, row, col, shovel_target, target_row, target_col);
    pvz::env::EventsTracker events;
    events.begin_step(before);

    result.action = timing.measure(T::Action, [&] { return pvz::action::ActionExecutor::execute(action); });

    // Fast headless treats the requested tick count as a target chunk size, not
    // an exact contract. A frame that was already in flight when the transient
    // block was armed may cause a small +/- clock delta. Real game_clock is the
    // authority and any positive progress is accepted.
    int advanced = 0;
    bool advance_failed = false;
    int advance_error = 0;

    result.automation_collected =
        std::max(0, timing.measure(T::Automation, [&] { return pvz::automation::AutomationController::update(); }));
    result.status = pvz::game::RuntimePhaseReader::read().status;

    // Zero progress can legitimately happen while an early tutorial state is
    // being cleared. Retry only zero-progress transitions after servicing the
    // automation layer. Once any positive clock delta is observed, return it as
    // the transition's actual duration even if it differs from `ticks`.
    constexpr int kMaxNoProgressRetries = 4;
    for (int attempt = 0;
         attempt < kMaxNoProgressRetries && result.status == pvz::GameStatus::Playing;
         ++attempt) {
        const int delta = timing.measure(T::Tick, [&] { return pvz::game::SimulationClock::advance_ticks_headless(ticks); });
        if (delta < 0) {
            advance_failed = true;
            advance_error = delta;
            break;
        }
        if (delta > 0) {
            advanced = delta;
            break;
        }

        result.automation_collected +=
            std::max(0, timing.measure(T::Automation, [&] { return pvz::automation::AutomationController::update(); }));
        result.status = pvz::game::RuntimePhaseReader::read().status;
    }

    result.automation_collected +=
        std::max(0, timing.measure(T::Automation, [&] { return pvz::automation::AutomationController::update(); }));
    result.status = pvz::game::RuntimePhaseReader::read().status;

    if (advance_failed) {
        result.error = pvz::ActionError::ActionTimeout;
        result.message =
            "headless tick advance failed with code " + std::to_string(advance_error);
    } else if (result.status == pvz::GameStatus::Playing && advanced <= 0) {
        result.error = pvz::ActionError::ActionTimeout;
        result.message =
            "headless tick advance made no progress for requested " +
            std::to_string(ticks) + " ticks";
    }

    pvz::Observation after;
    const bool fresh = timing.measure(T::After, [&] { return pvz::observation::ObservationBuilder::build(after); });
    const auto terminal_state = fresh ? pvz::game::RuntimeState{} : pvz::game::RuntimePhaseReader::read();
    if (!pvz::env::prepare_transition_observation(result, before, after, fresh, terminal_state,
                                                 before.global.game_clock))
        return publish(result, out_result_json, max_len);

    result.observation = after;
    result.events = events.end_step(after, action, result.action, result.status);

    if (result.status == pvz::GameStatus::Playing) {
        timing.measure(T::Mask, [&] { pvz::action::ActionMask::compute_mask(result.action_mask); });
    } else {
        result.action_mask.assign(pvz::action::ActionCodec::kActionSpaceSize, false);
    }

    if (!advance_failed &&
        (result.status != pvz::GameStatus::Playing || advanced > 0)) {
        result.success = true;
        result.error = pvz::ActionError::None;
        result.message =
            "headless step completed: requested=" + std::to_string(ticks) +
            " actual=" + std::to_string(advanced);
    } else {
        result.success = false;
    }

    return publish(result, out_result_json, max_len);
    } catch (const PvzRuntime::NativeOperationError& e) {
        pvz::StepResult err;
        err.error = e.error();
        err.message = e.what();
        return publish(err, out_result_json, max_len);
    } catch (...) {
        pvz::StepResult err;
        err.error = pvz::ActionError::Unsupported;
        err.message = "internal exception during headless step";
        return publish(err, out_result_json, max_len);
    }
}

}  // extern "C"
