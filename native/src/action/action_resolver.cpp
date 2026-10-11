#include "action_resolver.h"

#include "../game/rules/cob_rules.h"
#include "../game/rules/plant_rules.h"
#include "../game/rules/shovel_rules.h"
#include "../game/original/vase_calls.h"
#include "../process/process_manager.h"

namespace pvz::action {

ActionResolution ActionResolver::resolve(const Action& action) {
    {
        auto& mgr = process::ProcessManager::instance();
        std::lock_guard<std::mutex> lock(mgr.mutex());
        mgr.runtime().ensure_remote_idle();
    }
    ActionResolution out;

    switch (action.type) {
    case ActionType::NoOp:
        if (process::ProcessManager::instance().is_battle_ready()) {
            out.legal = true;
            out.error = ActionError::None;
            out.source = game::rules::RuleSource::OriginalState;
            out.payload = NoOpResolution{};
        } else {
            out.error = ActionError::GameNotInBattleState;
            out.source = game::rules::RuleSource::OriginalState;
        }
        return out;

    case ActionType::Plant: {
        const auto r = game::rules::PlantRules::resolve(action.card_slot, action.row, action.col);
        out.legal = r.decision.legal;
        out.error = r.decision.error;
        out.source = r.decision.source;
        out.payload = PlantResolution{action.card_slot, action.row, action.col};
        return out;
    }

    case ActionType::Shovel: {
        const auto r = game::rules::ShovelRules::resolve(
            action.row, action.col, action.shovel_target);
        out.legal = r.decision.legal;
        out.error = r.decision.error;
        out.source = r.decision.source;
        out.payload = ShovelResolution{
            action.row, action.col, action.shovel_target, r.target_plant_id};
        return out;
    }

    case ActionType::CobFire: {
        if (action.row != -1 || action.col != -1) {
            out.error = ActionError::InvalidArgument;
            return out;
        }
        const auto r = game::rules::CobRules::resolve_next(
            action.target_row, action.target_col);
        out.legal = r.decision.legal;
        out.error = r.decision.error;
        out.source = r.decision.source;
        out.payload = CobFireResolution{
            action.row, action.col, action.target_row, action.target_col,
            r.cannon_id, r.target_x, r.target_y, r.cannon_identity};
        return out;
    }
    case ActionType::InteractCell: {
        // Vasebreaker only; hammer interaction is intentionally deferred.
        const auto r = game::original::VaseCalls::resolve(action.row, action.col);
        out.legal = r.accepted;
        out.error = r.error;
        out.source = r.available ? game::rules::RuleSource::OriginalState : game::rules::RuleSource::Unsupported;
        out.payload = VaseResolution{action.row, action.col, r.identity, r.board};
        return out;
    }
    }

    out.error = ActionError::InvalidArgument;
    return out;
}

}  // namespace pvz::action
