#pragma once

#include "pvz/types.h"

#include <string>
#include <vector>

namespace pvz::game::original {

struct OriginalPlantCallResult {
    bool available = false;
    bool accepted = false;
    ActionError error = ActionError::Unsupported;
};

// Planting stack with a shared card-source reader. The original CanPlantAt
// supplies spatial legality for selected conveyor levels; other levels retain
// the hand-written rule while both results are compared. Flat-mask queries
// batch the original calls into one remote injection.
class PlantCalls {
public:
    static OriginalPlantCallResult query_legality(int slot, int row, int col);

    // IMPORTANT: commit() currently ends in the game's low-level
    // call_put_plant routine. That routine directly creates/places the Plant on
    // the Board; it is NOT equivalent to the complete player planting flow:
    //
    //   click SeedPacket -> pick up into Cursor -> Board::MouseDownWithPlant()
    //
    // Therefore call_put_plant does not automatically reproduce every
    // interaction-side effect of a real player click. In particular, tutorial
    // transitions such as Adventure 1-1's
    // PICK_UP_PEASHOOTER -> PLANT_PEASHOOTER state change can be bypassed.
    // Treat this path as a low-level placement primitive until the Agent Plant
    // action is migrated to the original player interaction path.
    static OriginalPlantCallResult commit(int slot, int row, int col);

    // Returns exactly 10 * 6 * 9 Plant bits with source and spatial gates.
    static bool build_flat_action_mask_shadow(std::vector<bool>& out_mask);

    // Runtime diagnostics for deciding when it is safe to retire the legacy
    // memory-heavy validator and promote Board::CanPlantAt to authority.
    static std::string comparison_stats_json();
    static void reset_comparison_stats();
};

}  // namespace pvz::game::original
