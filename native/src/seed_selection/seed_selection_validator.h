#pragma once

#include "pvz/seed_selection.h"

#include <vector>

namespace pvz::seed_selection {

class SeedSelectionValidator {
public:
    static SeedSelectionError can_select(int plant_type, const SeedSelectionState& state);
    static SeedSelectionError can_select_imitater(int target_type, const SeedSelectionState& state);
    static SeedSelectionError can_deselect(int plant_type, const SeedSelectionState& state);
    static SeedSelectionError can_confirm(const SeedSelectionState& state);

    // mask layout: [select 0..63][imitater target 0..63][deselect 0..63][confirm]
    static std::vector<bool> build_action_mask(
        const SeedSelectionState& state,
        int seed_type_capacity = kSeedTypeCapacity);
};

}  // namespace pvz::seed_selection
