#pragma once

#include "rule_types.h"
#include "pvz/types.h"

namespace pvz::game::rules {

struct ShovelRuleResult {
    RuleDecision decision;
    int target_plant_id = -1;
};

class ShovelRules {
public:
    // NORMAL follows original TOPPLANT_DIGGING_ORDER; PUMPKIN targets only the shell.
    static ShovelRuleResult resolve(int row, int col, ShovelTarget target);
};

}  // namespace pvz::game::rules
