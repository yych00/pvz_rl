#include "plant_rules.h"

#include "../original/plant_calls.h"

namespace pvz::game::rules {

PlantRuleResult PlantRules::resolve(int slot, int row, int col) {
    const auto native = original::PlantCalls::query_legality(slot, row, col);
    PlantRuleResult out;
    out.decision.legal = native.accepted;
    out.decision.error = native.error;
    out.decision.source = native.available ? RuleSource::FaithfulReplica : RuleSource::Unsupported;
    return out;
}

}  // namespace pvz::game::rules
