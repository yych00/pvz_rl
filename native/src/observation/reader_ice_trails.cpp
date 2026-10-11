#include "reader_internal.h"
#include <algorithm>

namespace pvz::observation {

bool read_ice_trails_nolock(PvzRuntime::VersionedPvZ& pvz, uintptr_t board, int grid_rows, std::vector<IceTrailState>& out_ice_trails) {
    out_ice_trails.clear();
    if (!board || grid_rows <= 0) return false;

    // 计算版本差值：原版 sun 偏移为 0x5560，年度版为 0x5578。
    uintptr_t board_var_offset = pvz.data().sun - 0x5560;
    uintptr_t ice_base = board + pvz.data().ice_trail_cd;

    // 草坪共 9 列，每列约 80 像素；min_x 记录冰车冰道推进到的最左像素坐标。
    // 对应行计时器大于 0 时，说明该行仍存在冰道。
    for (int r = 0; r < grid_rows; ++r) {
        int timer = pvz.read_memory_checked<int>({ice_base + r * 4});
        if (timer > 0) {
            int min_x = pvz.read_memory_checked<int>({ice_base + r * 4 - 24});

            // 根据冰道最左端像素坐标，换算其覆盖的网格列区间。
            int col_start = 8;
            for (int c = 0; c < 9; ++c) {
                int grid_right_x = 80 + (c + 1) * 80;
                if (grid_right_x > min_x) {
                    col_start = c;
                    break;
                }
            }
            int col_end = 8; // 冰道从最右侧草坪向左延伸。

            IceTrailState item;
            item.row = r;
            item.col_start = col_start;
            item.col_end = col_end;
            item.melt_countdown = timer;

            out_ice_trails.push_back(item);
        }
    }

    return true;
}

}  // 命名空间 pvz::observation
