#include "action_executor.h"

#include "action_resolver.h"
#include "../game/original/cob_calls.h"
#include "../game/original/plant_calls.h"
#include "../game/original/shovel_calls.h"
#include "../game/original/vase_calls.h"

namespace pvz::action {
namespace {
ActionResult rejected(ActionError error, const char* message) {
    return {false, error, message};
}

ActionResult committed(ActionError error, const char* message) {
    return {error == ActionError::None, error, message};
}
}  // namespace

ActionResult ActionExecutor::execute(const Action& action) {
    const auto resolution = ActionResolver::resolve(action);
    if (!resolution.legal) {
        return rejected(resolution.error, "action rejected by authoritative resolver");
    }

    switch (action.type) {
    case ActionType::NoOp:
        return committed(ActionError::None, "noop");

    case ActionType::Plant: {
        const auto& r = std::get<PlantResolution>(resolution.payload);
        const auto commit = game::original::PlantCalls::commit(r.slot, r.row, r.col);
        return committed(commit.error, commit.accepted ? "plant committed" : "plant commit rejected");
    }

    case ActionType::Shovel: {
        const auto& r = std::get<ShovelResolution>(resolution.payload);
        const auto commit = game::original::ShovelCalls::commit_target(r.target_plant_id);
        return committed(commit.error, commit.accepted ? "shovel committed" : "shovel commit rejected");
    }

    case ActionType::CobFire: {
        const auto& r = std::get<CobFireResolution>(resolution.payload);
        const auto commit = game::original::CobCalls::fire(
            r.cannon_id, r.cannon_identity, r.target_x, r.target_y);
        return committed(commit.error, commit.accepted ? "cob fire committed" : "cob fire commit rejected");
    }
    case ActionType::InteractCell: {
        const auto& r = std::get<VaseResolution>(resolution.payload);
        const auto commit = game::original::VaseCalls::commit(r.row, r.col, r.identity, r.board);
        return committed(commit.error, commit.accepted ? "vase opening started" : "vase interaction rejected");
    }
    }

    return rejected(ActionError::InvalidArgument, "unknown action type");
}

ActionResult ActionExecutor::plant(int slot, int row, int col) {
    Action action;
    action.type = ActionType::Plant;
    action.card_slot = slot;
    action.row = row;
    action.col = col;
    return execute(action);
}

ActionResult ActionExecutor::shovel(int row, int col, ShovelTarget target) {
    Action action;
    action.type = ActionType::Shovel;
    action.row = row;
    action.col = col;
    action.shovel_target = target;
    return execute(action);
}

ActionResult ActionExecutor::cob_fire(int target_row, int target_col) {
    Action action;
    action.type = ActionType::CobFire;
    action.target_row = target_row;
    action.target_col = target_col;
    return execute(action);
}

}  // namespace pvz::action
