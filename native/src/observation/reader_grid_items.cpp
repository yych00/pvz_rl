#include "reader_internal.h"
#include <algorithm>

namespace pvz::observation {

bool read_grid_items_nolock(PvzRuntime::VersionedPvZ& pvz, uintptr_t board, int grid_rows, std::vector<GridItemState>& out_grid_items) {
    out_grid_items.clear();
    if (!board || grid_rows <= 0) return false;

    uintptr_t grid_item_arr = pvz.read_memory_checked<uintptr_t>({board + pvz.data().grid_item});
    int grid_item_max = pvz.read_memory_checked<int>({board + pvz.data().grid_item_count_max});
    if (grid_item_max < 0 || grid_item_max > 128 || (grid_item_max && !grid_item_arr))
        throw PvzRuntime::NativeOperationError(ActionError::Unsupported, "invalid grid item array");

    if (!grid_item_arr || grid_item_max <= 0) return true;

    for (int k = 0; k < grid_item_max; ++k) {
        uintptr_t addr = grid_item_arr + k * PvzRuntime::grid_item_struct_size(pvz);
        bool dead = pvz.read_memory_checked<bool>({addr + pvz.data().grid_item_dead});
        if (dead) continue;

        int gtype = pvz.read_memory_checked<int>({addr + pvz.data().grid_item_type});
        int grow = pvz.read_memory_checked<int>({addr + pvz.data().grid_item_row});
        int gcol = pvz.read_memory_checked<int>({addr + pvz.data().grid_item_col});
        int cd = pvz.read_memory_checked<int>({addr + 0x18}); // 剩余倒计时，单位为游戏刻。

        if (grow < 0 || grow >= grid_rows || gcol < 0 || gcol >= 9) continue;

        // 只提取训练需要的格子物件：1=墓碑，2=弹坑，3=梯子，7=花瓶，6/12=脑子。
        std::string type_str;
        int remaining_ticks = 0;

        if (gtype == 1) {
            type_str = "GRAVE";
            remaining_ticks = 0;
        } else if (gtype == 2) {
            type_str = "CRATER";
            remaining_ticks = std::max(0, cd); // 弹坑保留真实剩余时间。
        } else if (gtype == 7) {
            type_str = "VASE";
            remaining_ticks = 0;
        } else if (gtype == 3) {
            type_str = "LADDER";
            remaining_ticks = 0;
        } else if (gtype == 12 || gtype == 6) {
            type_str = "BRAIN";
            remaining_ticks = 0;
        } else {
            continue; // 其他格子物件暂不进入观测。
        }

        GridItemState item;
        item.type = type_str;
        item.row = grow;
        item.col = gcol;
        item.remaining_ticks = remaining_ticks;

        out_grid_items.push_back(item);
    }

    return true;
}

}  // 命名空间 pvz::observation
