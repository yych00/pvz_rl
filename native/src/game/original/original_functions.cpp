#include "original_functions.h"

namespace pvz::game::original {

std::uintptr_t OriginalFunctions::address(FunctionId) {
    // Address wiring is intentionally deferred until each function is verified.
    // Never add guessed fallback addresses here.
    return 0;
}

}  // namespace pvz::game::original
