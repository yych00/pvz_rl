#include "runtime/native_errors.h"
#include "pvz/api.h"
#include "pvz/env.h"

#include "serializer.h"
#include "transition_result_buffer.h"
#include "shared_env.h"
#include "wire_result.h"
#include "../automation/auto_collect.h"
#include "../automation/auto_whack.h"
#include "../automation/automation_controller.h"
#include "../automation/intro_skipper.h"
#include "../game/original/plant_calls.h"
#include "../game/rng_control.h"
#include "../game/runtime_phase.h"
#include "../game/simulation_clock.h"
#include "../observation/observation_builder.h"
#include "../process/process_manager.h"

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <vector>

namespace pvz::api {
PvzEnv& shared_env() {
    static pvz::PvzEnv env;
    return env;
}
}

namespace {
using pvz::api::shared_env;

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

int action_error_code(const pvz::ActionResult& result) {
    return static_cast<int>(result.error);
}

std::string runtime_state_to_json(const pvz::game::RuntimeState& state) {
    std::stringstream ss;
    ss << "{";
    ss << "\"process_available\":" << (state.process_available ? "true" : "false") << ",";
    ss << "\"raw_scene\":" << state.raw_scene << ",";
    ss << "\"raw_scene_name\":\""
       << pvz::game::RuntimePhaseReader::raw_scene_name(state.raw_scene) << "\",";
    ss << "\"phase\":" << static_cast<int>(state.phase) << ",";
    ss << "\"phase_name\":\""
       << pvz::game::RuntimePhaseReader::phase_name(state.phase) << "\",";
    ss << "\"game_status\":" << static_cast<int>(state.status) << ",";
    ss << "\"board_present\":" << (state.board_present ? "true" : "false") << ",";
    ss << "\"cutscene_present\":" << (state.cutscene_present ? "true" : "false") << ",";
    ss << "\"seed_choosing\":" << (state.seed_choosing ? "true" : "false") << ",";
    ss << "\"dave_talking\":" << (state.dave_talking ? "true" : "false") << ",";
    ss << "\"level_complete\":" << (state.level_complete ? "true" : "false") << ",";
    ss << "\"cutscene_time\":" << state.cutscene_time << ",";
    ss << "\"dave_message\":" << state.dave_message << ",";
    ss << "\"game_mode\":" << state.game_mode << ",";
    ss << "\"level_id\":" << state.level_id << ",";
    ss << "\"board_id\":" << state.board_address << ",";
    ss << "\"game_clock\":" << state.game_clock << ",";
    ss << "\"game_clock_available\":" << (state.game_clock_available ? "true" : "false") << ",";
    ss << "\"finished_adventure\":" << state.finished_adventure;
    ss << "}";
    return ss.str();
}
}  // namespace

extern "C" {

int PVZ_GYM_CALL pvz_gym_get_runtime_phase() {
    try { return static_cast<int>(pvz::game::RuntimePhaseReader::read().phase); }
    catch (const PvzRuntime::NativeOperationError& e) { return -static_cast<int>(e.error()); }
    catch (...) { return -static_cast<int>(pvz::ActionError::Unsupported); }
}

int PVZ_GYM_CALL pvz_gym_get_runtime_state_json(char* out_json_buf, int max_len) {
    try {
        return pvz::api::Serializer::copy_to_c_buffer(
            runtime_state_to_json(pvz::game::RuntimePhaseReader::read()), out_json_buf, max_len);
    } catch (...) {
        return -1;
    }
}

bool PVZ_GYM_CALL pvz_gym_pause() { return pvz::game::SimulationClock::pause(); }
bool PVZ_GYM_CALL pvz_gym_resume() { return pvz::game::SimulationClock::resume(); }
bool PVZ_GYM_CALL pvz_gym_is_paused() { return pvz::game::SimulationClock::is_paused(); }
bool PVZ_GYM_CALL pvz_gym_step_one_tick() { return pvz::game::SimulationClock::step_one_tick(); }
int PVZ_GYM_CALL pvz_gym_advance_ticks(int ticks) { return pvz::game::SimulationClock::advance_ticks(ticks); }
int64_t PVZ_GYM_CALL pvz_gym_get_game_clock() { return pvz::game::SimulationClock::get_game_clock(); }

int PVZ_GYM_CALL pvz_gym_can_execute_action(
    int action_type, int card_slot, int row, int col,
    int shovel_target, int target_row, int target_col) {
    try {
        auto& env = shared_env();
        if (!env.attach()) return static_cast<int>(pvz::ActionError::GameNotFound);
        pvz::ActionError error = pvz::ActionError::Unsupported;
        const bool legal = env.can_execute(
            make_action(action_type, card_slot, row, col, shovel_target, target_row, target_col), &error);
        return legal ? 0 : static_cast<int>(error);
    } catch (const PvzRuntime::NativeOperationError& e) {
        return static_cast<int>(e.error());
    } catch (...) {
        return static_cast<int>(pvz::ActionError::Unsupported);
    }
}

int PVZ_GYM_CALL pvz_gym_execute_action(
    int action_type, int card_slot, int row, int col,
    int shovel_target, int target_row, int target_col,
    char* out_msg_buf, int max_msg_len) {
    try {
        auto& env = shared_env();
        pvz::ActionResult result;
        if (!env.attach()) {
            result = {false, pvz::ActionError::GameNotFound, "PlantsVsZombies.exe was not found"};
        } else {
            result = env.execute_action(
                make_action(action_type, card_slot, row, col, shovel_target, target_row, target_col));
        }
        if (out_msg_buf && max_msg_len > 0) {
            pvz::api::Serializer::copy_to_c_buffer(result.message, out_msg_buf, max_msg_len);
        }
        return action_error_code(result);
    } catch (const PvzRuntime::NativeOperationError& e) {
        if (out_msg_buf && max_msg_len > 0) pvz::api::Serializer::copy_to_c_buffer(e.what(), out_msg_buf, max_msg_len);
        return static_cast<int>(e.error());
    } catch (...) {
        return static_cast<int>(pvz::ActionError::Unsupported);
    }
}

int PVZ_GYM_CALL pvz_gym_get_action_mask(
    uint8_t* out_mask_buf, int max_size, int* out_count) {
    try {
        auto& env = shared_env();
        if (!env.attach()) {
            if (out_count) *out_count = 0;
            return static_cast<int>(pvz::ActionError::GameNotFound);
        }
        const std::vector<bool> mask = env.get_action_mask();
        const int count = static_cast<int>(mask.size());
        if (out_count) *out_count = count;
        if (!out_mask_buf || max_size < count) return -count;
        for (int i = 0; i < count; ++i) out_mask_buf[i] = mask[i] ? 1 : 0;
        return 0;
    } catch (...) {
        if (out_count) *out_count = 0;
        return -1;
    }
}

int PVZ_GYM_CALL pvz_gym_get_plant_legality_compare_json(char* out_json_buf, int max_len) {
    return pvz::api::Serializer::copy_to_c_buffer(
        pvz::game::original::PlantCalls::comparison_stats_json(), out_json_buf, max_len);
}

void PVZ_GYM_CALL pvz_gym_reset_plant_legality_compare_stats() {
    pvz::game::original::PlantCalls::reset_comparison_stats();
}

int PVZ_GYM_CALL pvz_gym_get_observation_json(char* out_json_buf, int max_len) {
    try {
        if (!out_json_buf || max_len <= 0) return -1;
        pvz::Observation obs;
        if (!pvz::observation::ObservationBuilder::build(obs)) return 0;
        return pvz::api::Serializer::copy_to_c_buffer(
            pvz::api::Serializer::observation_to_json(obs), out_json_buf, max_len);
    } catch (...) {
        return -1;
    }
}

uint64_t PVZ_GYM_CALL pvz_gym_begin_result_request() {
    return pvz::api::transition_results.begin();
}

int PVZ_GYM_CALL pvz_gym_read_result_request(
    uint64_t request_id, char* out_json_buf, int max_len) {
    return pvz::api::transition_results.read(request_id, out_json_buf, max_len);
}

int PVZ_GYM_CALL pvz_gym_env_reset(
    int level_id, uint64_t seed, bool has_seed, int timeout_ms,
    char* out_obs_json, int max_obs_len, int64_t* out_clock) {
    try {
        pvz::ResetConfig config;
        config.level_id = level_id;
        config.seed = seed;
        config.has_seed = has_seed;
        config.auto_select_seeds = true;
        config.deterministic_pause = true;
        config.wait_ready_timeout_ms = timeout_ms;
        const pvz::ResetResult result = shared_env().reset(config);
        if (out_clock) *out_clock = result.game_clock;
        if (!result.success || !result.observation_available) return -1;
        return pvz::api::transition_results.publish(
            pvz::api::Serializer::observation_to_json(result.observation), out_obs_json, max_obs_len);
    } catch (...) {
        return -1;
    }
}

int PVZ_GYM_CALL pvz_gym_env_reset_v2(
    int level_id,
    int playthrough,
    uint64_t seed,
    bool has_seed,
    bool stop_at_seed_selection,
    bool auto_select_seeds,
    const int32_t* seed_types,
    int seed_count,
    bool deterministic_pause,
    int timeout_ms,
    char* out_result_json,
    int max_len) {
    try {
        pvz::ResetConfig config;
        config.level_id = level_id;
        config.playthrough = playthrough;
        config.seed = seed;
        config.has_seed = has_seed;
        config.stop_at_seed_selection = stop_at_seed_selection;
        config.auto_select_seeds = auto_select_seeds;
        config.deterministic_pause = deterministic_pause;
        config.wait_ready_timeout_ms = timeout_ms;
        if (seed_count < 0 || (seed_count > 0 && seed_types == nullptr)) {
            pvz::ResetResult invalid;
            invalid.success = false;
            invalid.error = pvz::ActionError::InvalidArgument;
            invalid.message = "invalid seed_types pointer/count";
            return pvz::api::transition_results.publish(
                pvz::api::wire_result(invalid), out_result_json, max_len);
        }
        if (seed_types && seed_count > 0) {
            config.seed_types.assign(seed_types, seed_types + seed_count);
        }

        const pvz::ResetResult result = shared_env().reset(config);
        return pvz::api::transition_results.publish(
            pvz::api::wire_result(result), out_result_json, max_len);
    } catch (const PvzRuntime::NativeOperationError& e) {
        pvz::ResetResult result;
        result.success = false;
        result.error = e.error();
        result.message = e.what();
        return pvz::api::transition_results.publish(
            pvz::api::wire_result(result), out_result_json, max_len);
    } catch (...) {
        return -1;
    }
}

int PVZ_GYM_CALL pvz_gym_env_wait_for_battle(
    bool deterministic_pause,
    int timeout_ms,
    char* out_result_json,
    int max_len) {
    try {
        const pvz::ResetResult result = shared_env().wait_for_battle(timeout_ms, deterministic_pause);
        return pvz::api::transition_results.publish(
            pvz::api::wire_result(result), out_result_json, max_len);
    } catch (const PvzRuntime::NativeOperationError& e) {
        pvz::ResetResult result;
        result.success = false;
        result.error = e.error();
        result.message = e.what();
        return pvz::api::transition_results.publish(
            pvz::api::wire_result(result), out_result_json, max_len);
    } catch (...) {
        return -1;
    }
}

int PVZ_GYM_CALL pvz_gym_env_step(
    int action_type, int card_slot, int row, int col,
    int shovel_target, int target_row, int target_col,
    int ticks, char* out_result_json, int max_len) {
    try {
        const pvz::StepResult result = shared_env().step(
            make_action(action_type, card_slot, row, col, shovel_target, target_row, target_col), ticks);
        return pvz::api::transition_results.publish(
            pvz::api::wire_result(result), out_result_json, max_len);
    } catch (const PvzRuntime::NativeOperationError& e) {
        pvz::StepResult result;
        result.success = false;
        result.error = e.error();
        result.message = e.what();
        return pvz::api::transition_results.publish(
            pvz::api::wire_result(result), out_result_json, max_len);
    } catch (...) {
        return -1;
    }
}

int PVZ_GYM_CALL pvz_gym_env_step_continuous(
    int action_type, int card_slot, int row, int col,
    int shovel_target, int target_row, int target_col,
    int target_ticks, char* out_result_json, int max_len) {
    try {
        const pvz::StepResult result = shared_env().step_continuous(
            make_action(action_type, card_slot, row, col, shovel_target, target_row, target_col),
            target_ticks);
        return pvz::api::transition_results.publish(
            pvz::api::wire_result(result), out_result_json, max_len);
    } catch (const PvzRuntime::NativeOperationError& e) {
        pvz::StepResult result;
        result.success = false;
        result.error = e.error();
        result.message = e.what();
        return pvz::api::transition_results.publish(
            pvz::api::wire_result(result), out_result_json, max_len);
    } catch (...) {
        return -1;
    }
}

int PVZ_GYM_CALL pvz_gym_get_rng_seed_support() {
    return static_cast<int>(pvz::game::RngControl::support());
}

void PVZ_GYM_CALL pvz_gym_set_automation_config(bool enabled, bool auto_dialogue, bool auto_collect) {
    auto cfg = pvz::automation::AutomationController::get_config();
    cfg.enabled = enabled;
    cfg.auto_dialogue = auto_dialogue;
    cfg.auto_collect = auto_collect;
    pvz::automation::AutomationController::set_config(cfg);
}

void PVZ_GYM_CALL pvz_gym_get_automation_config(bool* out_enabled, bool* out_auto_dialogue, bool* out_auto_collect) {
    const auto cfg = pvz::automation::AutomationController::get_config();
    if (out_enabled) *out_enabled = cfg.enabled;
    if (out_auto_dialogue) *out_auto_dialogue = cfg.auto_dialogue;
    if (out_auto_collect) *out_auto_collect = cfg.auto_collect;
}

void PVZ_GYM_CALL pvz_gym_set_auto_whack(bool enable) {
    auto cfg = pvz::automation::AutomationController::get_config();
    cfg.auto_whack = enable;
    pvz::automation::AutomationController::set_config(cfg);
}

bool PVZ_GYM_CALL pvz_gym_get_auto_whack() {
    return pvz::automation::AutomationController::get_config().auto_whack;
}

bool PVZ_GYM_CALL pvz_gym_handle_prebattle() {
    try {
        return pvz::automation::AutomationController::handle_prebattle();
    } catch (...) {
        return false;
    }
}
int PVZ_GYM_CALL pvz_gym_automation_collect() {
    try {
        return pvz::automation::AutoCollect::collect_all();
    } catch (...) {
        return -1;
    }
}
bool PVZ_GYM_CALL pvz_gym_automation_whack() {
    try {
        return pvz::automation::AutoWhack::whack_all() > 0;
    } catch (...) {
        return false;
    }
}
int PVZ_GYM_CALL pvz_gym_automation_tick() {
    try {
        return pvz::automation::AutomationController::update();
    } catch (...) {
        return -1;
    }
}

bool PVZ_GYM_CALL pvz_gym_set_window_always_on_top(bool enable) {
    return pvz::process::ProcessManager::instance().set_always_on_top(enable);
}
void PVZ_GYM_CALL pvz_gym_set_background_running(bool enable) {
    pvz::process::ProcessManager::instance().set_background_running(enable);
}
bool PVZ_GYM_CALL pvz_gym_set_cheat_keys(bool enable) {
    return pvz::process::ProcessManager::instance().set_cheat_keys(enable);
}
bool PVZ_GYM_CALL pvz_gym_get_cheat_keys() {
    return pvz::process::ProcessManager::instance().get_cheat_keys();
}
bool PVZ_GYM_CALL pvz_gym_send_key(uint32_t vk_code) {
    return pvz::process::ProcessManager::instance().send_key(vk_code);
}
bool PVZ_GYM_CALL pvz_gym_cancel_intro() {
    try {
        return pvz::automation::IntroSkipper::update();
    } catch (...) {
        return false;
    }
}
void PVZ_GYM_CALL pvz_gym_set_auto_cancel_intro(bool enable) {
    pvz::automation::IntroSkipper::set_enabled(enable);
}
bool PVZ_GYM_CALL pvz_gym_get_auto_cancel_intro() {
    return pvz::automation::IntroSkipper::enabled();
}
uintptr_t PVZ_GYM_CALL pvz_gym_get_dialog(int dialog_id) {
    try { return pvz::process::ProcessManager::instance().get_dialog(dialog_id); } catch (...) { return 0; }
}
bool PVZ_GYM_CALL pvz_gym_kill_dialog(int dialog_id) {
    try {
        return pvz::process::ProcessManager::instance().kill_dialog(dialog_id);
    } catch (...) {
        return false;
    }
}

void PVZ_GYM_CALL pvz_gym_set_no_crazy_dave(bool enable) {
    pvz::process::ProcessManager::instance().set_no_crazy_dave(enable);
}

bool PVZ_GYM_CALL pvz_gym_get_no_crazy_dave() {
    return pvz::process::ProcessManager::instance().get_no_crazy_dave();
}

void PVZ_GYM_CALL pvz_gym_set_auto_dialogue(bool enable) {
    auto cfg = pvz::automation::AutomationController::get_config();
    cfg.auto_dialogue = enable;
    pvz::automation::AutomationController::set_config(cfg);
}

bool PVZ_GYM_CALL pvz_gym_get_auto_dialogue() {
    return pvz::automation::AutomationController::get_config().auto_dialogue;
}

}  // extern "C"
