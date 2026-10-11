#pragma once

#include "pvz/types.h"
#include <cstdint>
#include <vector>

namespace pvz::game::original {

struct OriginalVaseCallResult {
    bool available = false;
    bool accepted = false;
    ActionError error = ActionError::Unsupported;
    std::uint32_t identity = 0;
    uintptr_t board = 0;
};

class VaseCalls {
public:
    static OriginalVaseCallResult resolve(int row, int col);
    static OriginalVaseCallResult commit(int row, int col, std::uint32_t identity, uintptr_t board);
    static bool build_flat_action_mask(std::vector<bool>& out_mask);
};

} // namespace pvz::game::original
