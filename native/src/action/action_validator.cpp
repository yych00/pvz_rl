#include "action_validator.h"

#include "action_resolver.h"

namespace pvz::action {

bool ActionValidator::can_execute(const Action& action, ActionError* out_error) {
    const auto resolution = ActionResolver::resolve(action);
    if (out_error) *out_error = resolution.error;
    return resolution.legal;
}

}  // namespace pvz::action
