#include "pvz/env.h"

#include "../action/action_codec.h"
#include "../action/action_executor.h"
#include "../action/action_mask.h"
#include "../action/action_validator.h"
#include "../automation/automation_controller.h"
#include "../automation/tutorial_skipper.h"
#include "../game/rng_control.h"
#include "../game/runtime_phase.h"
#include "../game/simulation_clock.h"
#include "../observation/observation_builder.h"
#include "../process/process_manager.h"
#include "../process/runtime.h"
#include "../seed_selection/seed_selection_executor.h"
#include "../seed_selection/seed_selection_reader.h"
#include "../seed_selection/seed_selection_validator.h"
#include "events_tracker.h"
#include "transition_observation.h"
#include "../runtime/native_runtime.h"
#include "../runtime/turbo_runtime.h"
#include "../runtime/resident_executor.h"
#include "../api/step_timing.h"

#include <algorithm>
#include <chrono>
#include <thread>

namespace pvz {
namespace {

using game::RuntimePhase;
using game::RuntimePhaseReader;

ResetResult reset_error(ActionError error, const std::string& message) {
    ResetResult result;
    result.success = false;
    result.error = error;
    result.message = message;
    result.status = RuntimePhaseReader::read().status;
    result.rng_seed_support = game::RngControl::support();
    return result;
}

bool terminal_phase(RuntimePhase phase) noexcept {
    return phase == RuntimePhase::LevelComplete ||
           phase == RuntimePhase::Award ||
           phase == RuntimePhase::Lost;
}

void cleanup_target_dialogs() {
    auto& mgr = process::ProcessManager::instance();
    // Close Almanac even before AwardScreen's asynchronous continue path.
    bool killed_pause = false;
    for (int dialog_id : {2, 3, 19, 17, 37}) {
        if (mgr.get_dialog(dialog_id)) {
            (void)mgr.kill_dialog(dialog_id);
            if (dialog_id == 19) {
                killed_pause = true;
            }
        }
    }
    if (killed_pause && game::SimulationClock::is_paused()) {
        (void)game::SimulationClock::resume();
    }
}

bool wait_until_seed_or_battle(int timeout_ms, game::RuntimeState& out_state) {
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(std::max(timeout_ms, 1));

    while (std::chrono::steady_clock::now() < deadline) {
        cleanup_target_dialogs();
        out_state = RuntimePhaseReader::read();
        if (out_state.phase == RuntimePhase::SeedSelection ||
            out_state.phase == RuntimePhase::Battle) {
            return true;
        }
        if (out_state.phase == RuntimePhase::Menu ||
            out_state.phase == RuntimePhase::Offline) {
            return false;
        }

        if (out_state.phase == RuntimePhase::LevelIntro ||
            out_state.phase == RuntimePhase::Dialogue) {
            if (automation::TutorialSkipper::enabled()) {
                (void)automation::TutorialSkipper::skip_current();
            }
            (void)automation::AutomationController::handle_prebattle();
        }
        if (runtime::ResidentExecutor::instance().owner()) {
            runtime::NativeRuntime::instance().service();
            const auto phase = RuntimePhaseReader::read();
            if (phase.raw_scene == 2 && phase.phase != RuntimePhase::SeedSelection)
                game::SimulationClock::advance_intro_ticks(!runtime::TurboRuntime::instance().active() || runtime::TurboRuntime::instance().fast_intro() ? 200 : 1);
        } else std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    out_state = RuntimePhaseReader::read();
    // 战前辅助函数可能在内部真实推进过场动画，执行时间超过外层截止时间。
    // 如果它已经成功推进到选卡或战斗阶段，就接受最终状态而不是误报超时。
    return out_state.phase == RuntimePhase::SeedSelection ||
           out_state.phase == RuntimePhase::Battle;
}

bool wait_until_battle(int timeout_ms, game::RuntimeState& out_state) {
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(std::max(timeout_ms, 1));

    while (std::chrono::steady_clock::now() < deadline) {
        cleanup_target_dialogs();
        out_state = RuntimePhaseReader::read();
        if (out_state.phase == RuntimePhase::Battle) return true;
        if (out_state.phase == RuntimePhase::Menu ||
            out_state.phase == RuntimePhase::Offline ||
            out_state.phase == RuntimePhase::Lost ||
            out_state.phase == RuntimePhase::LevelComplete ||
            out_state.phase == RuntimePhase::Award) {
            return false;
        }
        if (automation::TutorialSkipper::enabled()) {
            (void)automation::TutorialSkipper::skip_current();
        }
        (void)automation::AutomationController::handle_prebattle();
        if (runtime::ResidentExecutor::instance().owner()) {
            runtime::NativeRuntime::instance().service();
            const auto phase = RuntimePhaseReader::read();
            if (phase.raw_scene == 2 && phase.phase != RuntimePhase::SeedSelection)
                game::SimulationClock::advance_intro_ticks(!runtime::TurboRuntime::instance().active() || runtime::TurboRuntime::instance().fast_intro() ? 200 : 1);
        } else std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    out_state = RuntimePhaseReader::read();
    // 同理，阻塞式战前辅助函数可能刚好在调用方截止时间到达时进入战斗。
    return out_state.phase == RuntimePhase::Battle;
}

ActionError complete_seed_selection(const ResetConfig& config, std::string& message) {
    using seed_selection::SeedSelectionExecutor;
    using seed_selection::SeedSelectionReader;

    if (!config.seed_types.empty()) {
        for (int plant_type : config.seed_types) {
            if (plant_type == kImitaterType) {
                message = "seed_types cannot specify Imitater without a target; use seed-selection actions";
                return ActionError::InvalidArgument;
            }
            const auto selected = SeedSelectionExecutor::select_seed(plant_type);
            if (!selected.success) {
                message = selected.message;
                return ActionError::InvalidArgument;
            }
        }

        SeedSelectionState state;
        if (!SeedSelectionReader::read(state) || !state.can_confirm) {
            message = "requested seed_types do not form a confirmable deck";
            return ActionError::InvalidArgument;
        }
        const auto confirmed = SeedSelectionExecutor::confirm_seeds();
        if (!confirmed.success) {
            message = confirmed.message;
            return ActionError::GameNotInBattleState;
        }
        return ActionError::None;
    }

    if (!config.auto_select_seeds) {
        message = "seed selection requires an agent, fixed seed_types, or auto_select_seeds";
        return ActionError::GameNotInBattleState;
    }

    for (int iteration = 0; iteration <= kSeedTypeCapacity; ++iteration) {
        SeedSelectionState state;
        if (!SeedSelectionReader::read(state)) {
            message = "seed chooser became unavailable";
            return ActionError::GameNotInBattleState;
        }
        if (state.can_confirm) {
            const auto confirmed = SeedSelectionExecutor::confirm_seeds();
            if (!confirmed.success) {
                message = confirmed.message;
                return ActionError::GameNotInBattleState;
            }
            return ActionError::None;
        }

        auto option = std::find_if(
            state.options.begin(), state.options.end(),
            [](const SeedOptionState& value) {
                return value.plant_type != kImitaterType && value.available &&
                       value.unlocked && !value.selected && !value.disabled;
            });
        if (option == state.options.end()) {
            const auto imitater_target = std::find_if(
                state.options.begin(), state.options.end(),
                [&state](const SeedOptionState& value) {
                    return value.plant_type >= 0 &&
                           value.plant_type < kImitaterTargetCount &&
                           seed_selection::SeedSelectionValidator::can_select_imitater(
                               value.plant_type, state) == SeedSelectionError::None;
                });
            if (imitater_target != state.options.end()) {
                const auto selected = SeedSelectionExecutor::select_imitater(
                    imitater_target->plant_type);
                if (!selected.success) {
                    message = selected.message;
                    return ActionError::InvalidArgument;
                }
                continue;
            }
            message = "seed chooser has no legal selection and cannot confirm";
            return ActionError::GameNotInBattleState;
        }
        const auto selected = SeedSelectionExecutor::select_seed(option->plant_type);
        if (!selected.success) {
            message = selected.message;
            return ActionError::InvalidArgument;
        }
    }

    message = "seed-selection iteration limit exceeded";
    return ActionError::ActionTimeout;
}

}  // 匿名命名空间

class PvzEnv::Impl {
public:
    env::EventsTracker events;
    Observation last_observation;
    bool has_last_observation = false;
};

PvzEnv::PvzEnv() : impl_(std::make_unique<Impl>()) {}
PvzEnv::~PvzEnv() = default;
PvzEnv::PvzEnv(PvzEnv&&) noexcept = default;
PvzEnv& PvzEnv::operator=(PvzEnv&&) noexcept = default;

bool PvzEnv::attach() { return process::attach() && runtime::NativeRuntime::instance().prepare(); }
bool PvzEnv::is_game_on() const { return process::game_on(); }

ResetResult PvzEnv::reset(const ResetConfig& config) {
    runtime::CriticalSectionGuard reset_guard;
    runtime::NativeRuntime::instance().new_episode();
    impl_->events.reset();
    impl_->has_last_observation = false;

    if (config.level_id < 1 || config.level_id > 50) {
        return reset_error(ActionError::InvalidArgument, "Adventure level_id must be in [1, 50]");
    }
    if (!attach()) {
        return reset_error(ActionError::GameNotFound, "PlantsVsZombies.exe was not found");
    }
    if (!runtime::NativeRuntime::instance().begin_lifecycle())
        return reset_error(ActionError::ActionTimeout, "resident lifecycle handoff failed: " +
            runtime::NativeRuntime::instance().lifecycle_error());

    // 不能继承上一局终局阶段留下的冻结 Board。
    if (game::SimulationClock::is_paused()) {
        (void)game::SimulationClock::resume();
    }
    cleanup_target_dialogs();
    process::ProcessManager::instance().set_background_running(true);

    if (config.playthrough != 0) {
        const int code = process::set_adventure_playthrough(config.playthrough);
        if (code != 0) {
            return reset_error(ActionError::InvalidArgument, "invalid Adventure playthrough");
        }
    }

    ResetResult result;
    result.rng_seed_support = game::RngControl::support();
    if (config.has_seed) {
        if (result.rng_seed_support != RngSeedSupport::Full) {
            return reset_error(
                ActionError::UnsupportedGameVersion,
                "deterministic episode RNG is unsupported for this PvZ build");
        }
        if (!game::RngControl::set_episode_seed(config.seed)) {
            return reset_error(ActionError::InvalidArgument, "episode RNG seed could not be applied");
        }
        result.rng_seed_applied = true;
    }

    const auto initial_state = RuntimePhaseReader::read();
    if (initial_state.raw_scene == 4 && runtime::TurboRuntime::instance().active() &&
        !runtime::TurboRuntime::instance().fast_terminal()) {
        if (game::SimulationClock::advance_terminal_updates(1100) < 0)
            return reset_error(ActionError::ActionTimeout, "loss cutscene update failed");
        cleanup_target_dialogs();
    }
    if (initial_state.phase == RuntimePhase::Menu && !runtime::TurboRuntime::instance().active() &&
        runtime::NativeRuntime::instance().options().executor != runtime::ExecutorKind::Resident) {
        return reset_error(
            ActionError::GameNotInBattleState,
            "native reset cannot enter Adventure from MENU; start a normal level once first");
    }

    const bool terminal_start = terminal_phase(initial_state.phase);
    const int attempts = terminal_start ? 3 : 1;
    bool left_terminal = !terminal_start;
    for (int attempt = 0; attempt < attempts; ++attempt) {
        const int code = process::select_level(config.level_id);
        if (code != 0) {
            return reset_error(
                static_cast<ActionError>(code),
                code == static_cast<int>(ActionError::ActionTimeout)
                    ? "select_level timed out; game thread released" : "select_level failed");
        }
        cleanup_target_dialogs();
        if (!terminal_start) break;

        const auto departure_deadline = std::chrono::steady_clock::now() +
            std::chrono::milliseconds(std::min(std::max(config.wait_ready_timeout_ms, 1), 2000));
        while (std::chrono::steady_clock::now() < departure_deadline) {
            if (!terminal_phase(RuntimePhaseReader::read().phase)) {
                left_terminal = true;
                break;
            }
            if (runtime::ResidentExecutor::instance().owner()) runtime::NativeRuntime::instance().service();
            else std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (left_terminal) break;
    }
    if (!left_terminal) {
        return reset_error(ActionError::ActionTimeout, "level transition did not leave terminal UI");
    }

    game::RuntimeState state;
    if (!wait_until_seed_or_battle(config.wait_ready_timeout_ms, state)) {
        return reset_error(
            ActionError::ActionTimeout,
            "timed out waiting for seed selection or battle");
    }

    if (state.phase == RuntimePhase::SeedSelection && config.stop_at_seed_selection) {
        result.success = true;
        result.error = ActionError::None;
        result.message = "episode prepared at seed selection";
        result.status = GameStatus::ChoosingSeeds;
        result.observation_available = false;
        result.game_clock = 0;
        return result;
    }

    if (state.phase == RuntimePhase::SeedSelection) {
        std::string message;
        const auto error = complete_seed_selection(config, message);
        if (error != ActionError::None) return reset_error(error, message);
    }

    ResetResult battle = wait_for_battle(config.wait_ready_timeout_ms, config.deterministic_pause);
    battle.rng_seed_support = result.rng_seed_support;
    battle.rng_seed_applied = result.rng_seed_applied;
    return battle;
}

ResetResult PvzEnv::wait_for_battle(std::int32_t timeout_ms, bool deterministic_pause) {
    if (!attach()) return reset_error(ActionError::GameNotFound, "PlantsVsZombies.exe was not found");

    game::RuntimeState state;
    if (!wait_until_battle(timeout_ms, state)) {
        return reset_error(ActionError::ActionTimeout, "timed out waiting for battle");
    }

    cleanup_target_dialogs();
    if (automation::TutorialSkipper::enabled()) {
        (void)automation::TutorialSkipper::skip_current();
    }

    if (deterministic_pause) {
        // 沿用已经验证稳定的实时 reset 行为：先让进入战斗后的第一帧刷新掉陈旧终局画面，
        // 再冻结 Board::Update，确保确定性步进从有效战斗帧开始。
        if (runtime::ResidentExecutor::instance().owner()) {
            if (game::SimulationClock::advance_ticks(1) < 0)
                return reset_error(ActionError::ActionTimeout, "initial battle update failed");
            if (automation::TutorialSkipper::enabled()) (void)automation::TutorialSkipper::skip_current();
        } else std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (!game::SimulationClock::is_paused() && !game::SimulationClock::pause()) {
            return reset_error(ActionError::GameNotInBattleState, "cannot pause battle for deterministic stepping");
        }
        if (!game::SimulationClock::is_paused()) {
            return reset_error(ActionError::GameNotInBattleState, "battle did not enter paused state");
        }
    }

    Observation observation;
    if (!observation::ObservationBuilder::build(observation)) {
        return reset_error(ActionError::Unsupported, "battle observation is unavailable after reset");
    }

    impl_->last_observation = observation;
    impl_->has_last_observation = true;
    impl_->events.reset();

    ResetResult result;
    if (!runtime::NativeRuntime::instance().finish_lifecycle())
        return reset_error(ActionError::ActionTimeout, "resident lifecycle release failed");
    result.success = true;
    result.error = ActionError::None;
    result.message = "battle ready";
    result.observation = observation;
    result.observation_available = true;
    result.status = GameStatus::Playing;
    result.game_clock = observation.global.game_clock;
    result.rng_seed_support = game::RngControl::support();
    return result;
}

StepResult PvzEnv::step(const Action& action, std::int32_t ticks) {
    runtime::RuntimeSnapshot cached_before;
    const bool reuse_before = runtime::NativeRuntime::instance().read_cached_snapshot(cached_before);
    runtime::NativeRuntime::instance().invalidate();
    StepResult result;
    result.action = {false, ActionError::Unsupported, "transition not executed"};

    if (!attach()) {
        result.error = ActionError::GameNotFound;
        result.message = "PlantsVsZombies.exe was not found";
        return result;
    }
    if (ticks <= 0) {
        result.error = ActionError::InvalidArgument;
        result.message = "ticks must be positive";
        return result;
    }
    if (status() != GameStatus::Playing) {
        result.error = ActionError::GameNotInBattleState;
        result.message = "step requires an active battle";
        result.status = status();
        return result;
    }

    Observation before;
    if (reuse_before) {
        before = cached_before.observation;
    } else if (!api::measure_step(api::StepTiming::Before, [&] { return observation::ObservationBuilder::build(before); })) {
        result.error = ActionError::Unsupported;
        result.message = "pre-step observation is unavailable";
        return result;
    }
    impl_->events.begin_step(before);

    using T = api::StepTiming;
    result.action = api::measure_step(T::Action, [&] { return execute_action(action); });
    const int advanced = api::measure_step(T::Tick, [&] { return game::SimulationClock::advance_ticks(ticks); });
    result.automation_collected = std::max(0, api::measure_step(T::Automation, [&] { return automation::AutomationController::update(); }));
    result.status = status();

    if (advanced < 0 || (result.status == GameStatus::Playing && advanced != ticks)) {
        result.error = ActionError::ActionTimeout;
        result.message = "native tick advance did not complete the requested transition";
    }

    Observation after;
    const bool fresh = api::measure_step(T::After, [&] { return observation::ObservationBuilder::build(after); });
    const auto terminal_state = fresh ? game::RuntimeState{} : RuntimePhaseReader::read();
    if (!env::prepare_transition_observation(result, before, after, fresh, terminal_state,
                                             before.global.game_clock)) {
        impl_->events.reset();
        return result;
    }

    result.observation = after;
    result.events = impl_->events.end_step(after, action, result.action, result.status);

    if (result.status == GameStatus::Playing) {
        result.action_mask = api::measure_step(T::Mask, [&] { return get_action_mask(); });
    } else {
        result.action_mask.assign(action::ActionCodec::kActionSpaceSize, false);
    }

    impl_->last_observation = after;
    impl_->has_last_observation = true;

    if (result.error == ActionError::Unsupported && result.message.empty()) {
        // 默认初始化的 Unsupported 只在状态转移尚未执行前有意义。
        result.error = ActionError::None;
    }
    if (result.error == ActionError::None ||
        (result.error == ActionError::Unsupported && result.message.empty())) {
        result.success = true;
        result.error = ActionError::None;
        result.message = "step completed";
    } else if (result.error == ActionError::Unsupported && result.message == "transition not executed") {
        result.success = false;
    } else {
        result.success = false;
    }

    // 正常路径到这里时环境层没有错误。非法智能体动作只记录在 result.action，
    // 不应该让整个状态转移本身失效。
    if (advanced >= 0 && (result.status != GameStatus::Playing || advanced == ticks)) {
        result.success = true;
        result.error = ActionError::None;
        result.message = "step completed";
    }
    runtime::NativeRuntime::instance().cache_transition(result.observation, result.action_mask);
    return result;
}

StepResult PvzEnv::step_continuous(const Action& action, std::int32_t target_ticks) {
    StepResult result;
    result.action = {false, ActionError::Unsupported, "transition not executed"};

    if (!attach()) {
        result.error = ActionError::GameNotFound;
        result.message = "PlantsVsZombies.exe was not found";
        return result;
    }
    if (target_ticks <= 0) {
        result.error = ActionError::InvalidArgument;
        result.message = "target_ticks must be positive";
        return result;
    }

    Observation before;
    if (impl_->has_last_observation) {
        before = impl_->last_observation;
    } else if (!observation::ObservationBuilder::build(before)) {
        result.error = ActionError::Unsupported;
        result.message = "pre-step observation is unavailable";
        return result;
    }
    if (before.quality != "fresh" || before.global.status == GameStatus::Won ||
        before.global.status == GameStatus::Lost) {
        result.error = ActionError::GameNotInBattleState;
        result.message = "continuous terminal transition was already settled; reset is required";
        return result;
    }
    impl_->events.begin_step(before);
    std::int64_t last_confirmed_clock = before.global.game_clock;

    result.status = status();
    bool wait_completed = result.status != GameStatus::Playing;
    if (result.status == GameStatus::Playing) {
        // 连续模式不应该持有确定性暂停。如果此前步进或无头流程留下暂停状态，先安全恢复战斗。
        if (game::SimulationClock::is_paused() && !game::SimulationClock::resume()) {
            result.error = ActionError::GameNotInBattleState;
            result.message = "continuous step could not resume the battle";
        } else {
            result.action = execute_action(action);
            const std::int64_t action_clock = game::SimulationClock::get_game_clock();
            const std::int64_t target_clock =
                static_cast<std::int64_t>(before.global.game_clock) + target_ticks;
            const int timeout_ms = std::max(5000, target_ticks * 100 + 1000);
            const auto deadline = std::chrono::steady_clock::now() +
                std::chrono::milliseconds(timeout_ms);

            std::int64_t current_clock = action_clock;
            last_confirmed_clock = std::max(last_confirmed_clock, action_clock);
            if (current_clock >= target_clock) {
                wait_completed = true;
            }
            while (!wait_completed && std::chrono::steady_clock::now() < deadline) {
                const auto current_state = RuntimePhaseReader::read();
                result.status = current_state.status;
                if (current_state.game_clock_available &&
                    current_state.board_address == before.global.board_id)
                    last_confirmed_clock = std::max(last_confirmed_clock, current_state.game_clock);
                if (result.status != GameStatus::Playing) {
                    wait_completed = true;
                    break;
                }
                current_clock = game::SimulationClock::get_game_clock();
                last_confirmed_clock = std::max(last_confirmed_clock, current_clock);
                if (current_clock >= target_clock) {
                    wait_completed = true;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (!wait_completed && result.status == GameStatus::Playing) {
                result.error = ActionError::ActionTimeout;
                result.message = "continuous clock made insufficient progress";
            }
        }
    } else {
        // 策略计算可能耗时到下一次动作抵达前战斗已经结束。这仍是合法终局转移，
        // 只把迟到动作记录为动作层失败，不让环境层转移失败。
        result.action = {
            false,
            ActionError::GameNotInBattleState,
            "action arrived after the battle ended",
        };
    }

    result.automation_collected =
        std::max(0, automation::AutomationController::update());
    result.status = status();
    if (result.status != GameStatus::Playing) {
        wait_completed = true;
    }

    Observation after;
    const bool fresh = observation::ObservationBuilder::build(after);
    const auto terminal_state = fresh ? game::RuntimeState{} : RuntimePhaseReader::read();
    if (!env::prepare_transition_observation(result, before, after, fresh, terminal_state,
                                             last_confirmed_clock)) {
        impl_->events.reset();
        return result;
    }

    result.observation = after;
    result.events = impl_->events.end_step(after, action, result.action, result.status);
    if (result.status == GameStatus::Playing) {
        result.action_mask = get_action_mask();
    } else {
        result.action_mask.assign(action::ActionCodec::kActionSpaceSize, false);
    }
    impl_->last_observation = after;
    impl_->has_last_observation = true;

    if (wait_completed) {
        result.success = true;
        result.error = ActionError::None;
        result.message = "continuous step completed";
    } else {
        result.success = false;
    }
    return result;
}

GameStatus PvzEnv::status() const {
    return game::RuntimePhaseReader::read().status;
}

Observation PvzEnv::get_observation() const {
    Observation obs;
    if (!observation::ObservationBuilder::build(obs))
        throw PvzRuntime::NativeOperationError(ActionError::Unsupported, "observation is unavailable");
    return obs;
}

std::vector<bool> PvzEnv::get_action_mask() const {
    std::vector<bool> mask;
    action::ActionMask::compute_mask(mask);
    return mask;
}

ActionResult PvzEnv::execute_action(const Action& action_value) {
    return action::ActionExecutor::execute(action_value);
}

bool PvzEnv::can_execute(const Action& action_value, ActionError* out_error) const {
    return action::ActionValidator::can_execute(action_value, out_error);
}

bool PvzEnv::pause() { return game::SimulationClock::pause(); }
bool PvzEnv::resume() { return game::SimulationClock::resume(); }
bool PvzEnv::is_paused() const { return game::SimulationClock::is_paused(); }

bool PvzEnv::advance_ticks(std::int32_t ticks) {
    return game::SimulationClock::advance_ticks(ticks) >= 0;
}

std::int64_t PvzEnv::get_game_clock() const {
    return game::SimulationClock::get_game_clock();
}

}  // 命名空间 pvz
