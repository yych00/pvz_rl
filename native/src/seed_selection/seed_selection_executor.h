#pragma once

#include "pvz/seed_selection.h"

namespace pvz::seed_selection {

class SeedSelectionExecutor {
public:
    static SeedSelectionResult select_seed(int plant_type);
    static SeedSelectionResult select_imitater(int target_type);
    static SeedSelectionResult deselect_seed(int plant_type);
    static SeedSelectionResult confirm_seeds();
};

}  // namespace pvz::seed_selection
