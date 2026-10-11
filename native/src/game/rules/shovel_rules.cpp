#include "shovel_rules.h"

#include "../original/shovel_calls.h"

namespace pvz::game::rules {

ShovelRuleResult ShovelRules::resolve(int row, int col, ShovelTarget target) {
    const auto native = original::ShovelCalls::resolve_target(row, col, target);
    ShovelRuleResult out;
    out.decision.legal = native.accepted;
    out.decision.error = native.error;
    out.decision.source = native.available ? RuleSource::FaithfulReplica : RuleSource::Unsupported;
    out.target_plant_id = native.target_plant_id;
    return out;
}

}  // namespace pvz::game::rules
