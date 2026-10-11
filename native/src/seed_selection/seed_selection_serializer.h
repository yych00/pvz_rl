#pragma once

#include "pvz/seed_selection.h"

#include <string>

namespace pvz::seed_selection {

class SeedSelectionSerializer {
public:
    static std::string to_json(const SeedSelectionState& state);
};

}  // namespace pvz::seed_selection
