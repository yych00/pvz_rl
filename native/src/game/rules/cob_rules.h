#pragma once

#include "rule_types.h"
#include <cstdint>

namespace pvz::game::rules {

struct CobRuleResult {
    RuleDecision decision;
    int cannon_id = -1;
    int target_x = -1;
    int target_y = -1;
    std::uint32_t cannon_identity = 0;
};

class CobRules {
public:
    static CobRuleResult resolve(
        int source_row,
        int source_col,
        int target_row,
        int target_col);
    static CobRuleResult resolve_next(int target_row, int target_col);
};

}  // namespace pvz::game::rules
