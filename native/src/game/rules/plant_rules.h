#pragma once

#include "rule_types.h"

namespace pvz::game::rules {

struct PlantRuleResult {
    RuleDecision decision;
};

class PlantRules {
public:
    static PlantRuleResult resolve(int slot, int row, int col);
};

}  // namespace pvz::game::rules
