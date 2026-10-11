#include "reader_internal.h"
#include <algorithm>

namespace pvz::observation {

bool read_mowers_nolock(PvzRuntime::VersionedPvZ& pvz, uintptr_t board, int grid_rows, std::vector<MowerState>& out_mowers) {
    out_mowers.clear();
    if (!board || grid_rows <= 0) return false;

    out_mowers.resize(grid_rows);
    for (int r = 0; r < grid_rows; ++r) {
        out_mowers[r].row = r;
    }

    uintptr_t mower_arr = pvz.read_memory_checked<uintptr_t>({board + pvz.data().lawn_mower});
    int mower_max = pvz.read_memory_checked<int>({board + pvz.data().lawn_mower_count_max});
    if (mower_max < 0 || mower_max > 32 || (mower_max && !mower_arr))
        throw PvzRuntime::NativeOperationError(ActionError::Unsupported, "invalid mower array");

    if (!mower_arr || mower_max <= 0) return true;

    for (int m = 0; m < mower_max; ++m) {
        uintptr_t addr = mower_arr + m * 0x48;
        bool dead = ((pvz.read_memory_checked<int>({addr + 0x30}) & 0xFF) != 0);
        int mrow = pvz.read_memory_checked<int>({addr + 0x14});
        // +0x18 is animation timing, not state. Original enum at +0x2C:
        // 0=rolling in, 1=ready, 2=triggered, 3=squished.
        int mstate = pvz.read_memory_checked<int>({addr + 0x2C});
        if (dead && mstate != 3) continue;

        if (mrow >= 0 && mrow < grid_rows) {
            if (mstate == 3) {
                out_mowers[mrow].combat_destroyed = true;
            } else if (mstate == 1 && !dead) {
                out_mowers[mrow].state = "READY";
            } else if (mstate == 2 && !dead) {
                out_mowers[mrow].state = "ROLLING";
            } else {
                out_mowers[mrow].state = "LOST";
            }
        }
    }

    return true;
}

}  // 命名空间 pvz::observation
