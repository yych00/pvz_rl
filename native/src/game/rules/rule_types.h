#pragma once

#include "pvz/types.h"

namespace pvz::game::rules {

enum class RuleSource {
    OriginalFunction,
    OriginalState,
    FaithfulReplica,
    Unsupported,
};

struct RuleDecision {
    bool legal = false;
    ActionError error = ActionError::Unsupported;
    RuleSource source = RuleSource::Unsupported;
};

}  // namespace pvz::game::rules
