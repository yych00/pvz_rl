#pragma once

#include "pvz/types.h"
#include <stdexcept>
#include <string>

namespace PvzRuntime {
class NativeOperationError : public std::runtime_error {
public:
    NativeOperationError(pvz::ActionError error, const std::string& message)
        : std::runtime_error(message), error_(error) {}
    pvz::ActionError error() const noexcept { return error_; }
private:
    pvz::ActionError error_;
};
}  // namespace PvzRuntime
