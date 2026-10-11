#pragma once

#include <cstdint>

namespace pvz::game::original {

enum class FunctionId {
    PlantLegality,
    PlantCommit,
    ShovelResolveTarget,
    ShovelCommit,
    CobFire,
};

class OriginalFunctions {
public:
    // Returns a verified address for the currently supported game build, or 0.
    static std::uintptr_t address(FunctionId id);
};

}  // namespace pvz::game::original
