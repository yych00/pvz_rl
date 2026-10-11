#pragma once

#include "pvz/types.h"
#include "../game/runtime_phase.h"

#include <algorithm>

namespace pvz::env {

// A fallback communicates a known terminal outcome, not newly read entity or
// resource facts. Duration is only the last clock actually observed on the
// original Board; requested ticks and Board::Update calls are not substitutes.
inline bool prepare_transition_observation(StepResult& result,
                                           const Observation& before,
                                           Observation& after, bool fresh,
                                           const game::RuntimeState& state,
                                           std::int64_t last_confirmed_clock) {
    if (fresh) {
        if ((before.global.board_id && after.global.board_id != before.global.board_id) ||
            after.global.game_clock < before.global.game_clock || after.quality != "fresh") {
            result.error = ActionError::Unsupported;
            result.message = "transition Board changed or observation clock regressed";
            return false;
        }
        result.status = after.global.status;
        result.game_clock = after.global.game_clock;
        result.last_confirmed_game_clock = after.global.game_clock;
        result.ticks_advanced = after.global.game_clock - before.global.game_clock;
        result.ticks_advanced_exact = true;
        result.replay_eligible = true;
        result.observation_quality = "fresh";
        return true;
    }
    // A required-reader false/exception while Board is still present is an
    // error, even in Lost/Award. Fallback is reserved for released terminal Board.
    if (!state.process_available || state.board_present ||
        (state.status != GameStatus::Won && state.status != GameStatus::Lost)) {
        result.error = ActionError::Unsupported;
        result.message = "post-step observation is unavailable without a released terminal Board";
        return false;
    }
    after = before;
    after.quality = "terminal_fallback";
    after.global.status = state.status;
    after.global.raw_scene = state.raw_scene;
    after.global.runtime_phase = static_cast<std::int32_t>(state.phase);
    // Keep the snapshot clock intact. StepResult separately exposes the latest
    // observed clock, which may be newer than these fallback entities.
    result.status = state.status;
    result.game_clock = std::max(before.global.game_clock, last_confirmed_clock);
    result.last_confirmed_game_clock = result.game_clock;
    result.ticks_advanced = result.game_clock - before.global.game_clock;
    result.ticks_advanced_exact = false;
    result.replay_eligible = false;
    result.observation_quality = after.quality;
    result.terminal_observation_fallback = true;
    return true;
}

} // namespace pvz::env
