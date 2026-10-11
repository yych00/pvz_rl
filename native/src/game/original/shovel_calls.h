#pragma once

#include "pvz/types.h"

#include <vector>

namespace pvz::game::original {

struct OriginalShovelTargetResult {
    bool available = false;
    bool accepted = false;
    ActionError error = ActionError::Unsupported;
    int target_plant_id = -1;
};

class ShovelCalls {
public:
    // NORMAL: original digging order (main plant, otherwise carrier).
    // PUMPKIN: only the pumpkin shell, matching a click on its visible rim.
    static OriginalShovelTargetResult resolve_target(int row, int col, ShovelTarget target);
    static OriginalShovelTargetResult commit_target(int target_plant_id);

    // Build 108 bits (54 normal, then 54 pumpkin) with one lock and one scan.
    static bool build_flat_action_mask(std::vector<bool>& out_mask);
};

}  // namespace pvz::game::original
