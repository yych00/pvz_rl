#pragma once

#include "pvz/types.h"

#include <cstdint>
#include <vector>

namespace pvz::game::original {

struct OriginalCobCallResult {
    bool available = false;
    bool accepted = false;
    ActionError error = ActionError::Unsupported;
    int cannon_id = -1;
    int target_x = -1;
    int target_y = -1;
    std::uint32_t cannon_identity = 0;
};

class CobCalls {
public:
    // Public contract is discrete grid targeting. This method converts the
    // target cell to the original Board pixel coordinate inside C++.
    static OriginalCobCallResult resolve(
        int source_row, int source_col, int target_row, int target_col);
    static OriginalCobCallResult resolve_next(int target_row, int target_col);
    static OriginalCobCallResult fire(
        int cannon_id, std::uint32_t cannon_identity, int target_x, int target_y);

    // Build 54 target bits with one process lock and one plant-array scan.
    static bool build_flat_action_mask(std::vector<bool>& out_mask);
};

}  // namespace pvz::game::original
