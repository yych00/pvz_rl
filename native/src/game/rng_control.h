#pragma once

#include "pvz/types.h"

#include <cstdint>

namespace pvz::game {

class RngControl {
public:
    // Episode RNG control is deliberately version-gated. Unsupported builds
    // must report Unsupported rather than guessing a LawnApp field offset.
    static RngSeedSupport support();
    static bool set_episode_seed(std::uint64_t seed);
};

}  // namespace pvz::game
