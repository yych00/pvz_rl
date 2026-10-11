#include "cob_rules.h"

#include "../original/cob_calls.h"

namespace pvz::game::rules {

CobRuleResult CobRules::resolve(int source_row, int source_col,
                                int target_row, int target_col) {
    const auto native = original::CobCalls::resolve(
        source_row, source_col, target_row, target_col);
    CobRuleResult out;
    out.decision.legal = native.accepted;
    out.decision.error = native.error;
    out.decision.source = native.available ? RuleSource::OriginalState : RuleSource::Unsupported;
    out.cannon_id = native.cannon_id;
    out.target_x = native.target_x;
    out.target_y = native.target_y;
    out.cannon_identity = native.cannon_identity;
    return out;
}

CobRuleResult CobRules::resolve_next(int target_row, int target_col) {
    const auto native = original::CobCalls::resolve_next(target_row, target_col);
    CobRuleResult out;
    out.decision.legal = native.accepted;
    out.decision.error = native.error;
    out.decision.source = native.available ? RuleSource::OriginalState : RuleSource::Unsupported;
    out.cannon_id = native.cannon_id;
    out.target_x = native.target_x;
    out.target_y = native.target_y;
    out.cannon_identity = native.cannon_identity;
    return out;
}

}  // namespace pvz::game::rules
