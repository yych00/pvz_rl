#include "reader_internal.h"
#include <algorithm>

namespace pvz::observation {

bool read_grid_nolock(PvzRuntime::VersionedPvZ& pvz, uintptr_t board, int scene, int grid_rows,
                      const std::vector<PlantState>& plants, std::vector<GridSquareState>& out_grid) {
    out_grid.clear();
    if (!board || grid_rows <= 0) return false;

    // 1. 初始化“行数×9”的网格数组。
    out_grid.reserve(grid_rows * 9);
    for (int r = 0; r < grid_rows; ++r) {
        for (int c = 0; c < 9; ++c) {
            GridSquareState sq;
            sq.row = r;
            sq.col = c;

            // 地形编号：0=草地，1=泳池水路，2=屋顶斜坡，3=屋顶平地。
            int terrain = 0;
            if (scene == 2 || scene == 3) { // 白天/夜晚泳池场景。
                if (r == 2 || r == 3) {
                    terrain = 1; // 中间两行为泳池水路。
                } else {
                    terrain = 0; // 上下为草坪。
                }
            } else if (scene == 4 || scene == 5) { // 屋顶场景。
                if (c < 4) {
                    terrain = 2; // 前四列为屋顶斜坡。
                } else {
                    terrain = 3; // 后五列为屋顶平地。
                }
            } else {
                terrain = 0; // 白天/黑夜普通草地。
            }
            sq.terrain = terrain;
            sq.has_base_plant = false;
            sq.has_main_plant = false;
            sq.has_pumpkin = false;
            sq.has_grave = false;
            sq.has_crater = false;

            out_grid.push_back(sq);
        }
    }

    // 2. 使用已经读取的植物回填格子占用状态，避免再次遍历植物内存。
    auto get_sq = [&](int r, int c) -> GridSquareState* {
        if (r >= 0 && r < grid_rows && c >= 0 && c < 9) {
            return &out_grid[r * 9 + c];
        }
        return nullptr;
    };

    for (const auto& p : plants) {
        GridSquareState* sq = get_sq(p.row, p.col);
        if (!sq) continue;

        if (p.layer == 0) {
            sq->has_base_plant = true; // 睡莲或花盆。
        } else if (p.layer == 2) {
            sq->has_pumpkin = true;    // 南瓜头。
        } else {
            sq->has_main_plant = true; // 主植物。
            // 玉米加农炮横跨两列，因此右侧底座格也标记为主体占用。
            if (p.type == 47) {
                GridSquareState* sq_right = get_sq(p.row, p.col + 1);
                if (sq_right) {
                    sq_right->has_main_plant = true;
                }
            }
        }
    }

    // 3. 读取格子物件数组，补充墓碑和弹坑状态。
    uintptr_t grid_item_arr = pvz.read_memory_checked<uintptr_t>({board + pvz.data().grid_item});
    int grid_item_max = pvz.read_memory_checked<int>({board + pvz.data().grid_item_count_max});
    if (grid_item_max < 0 || grid_item_max > 128 || (grid_item_max && !grid_item_arr))
        throw PvzRuntime::NativeOperationError(ActionError::Unsupported, "invalid grid item array");

    if (grid_item_arr && grid_item_max > 0) {
        for (int k = 0; k < grid_item_max; ++k) {
            uintptr_t addr = grid_item_arr + k * PvzRuntime::grid_item_struct_size(pvz);
            bool dead = pvz.read_memory_checked<bool>({addr + pvz.data().grid_item_dead});
            if (dead) continue;

            int gtype = pvz.read_memory_checked<int>({addr + pvz.data().grid_item_type});
            int grow = pvz.read_memory_checked<int>({addr + pvz.data().grid_item_row});
            int gcol = pvz.read_memory_checked<int>({addr + pvz.data().grid_item_col});

            GridSquareState* sq = get_sq(grow, gcol);
            if (!sq) continue;

            if (gtype == 1) { // 墓碑。
                sq->has_grave = true;
            } else if (gtype == 2) { // 弹坑。
                sq->has_crater = true;
            }
        }
    }

    return true;
}

}  // 命名空间 pvz::observation
