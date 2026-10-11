#pragma once

#include "pvz/types.h"

namespace pvz::observation {

// 只读事实快照总装器 (零副作用提取游戏状态)
class ObservationBuilder {
public:
    // 获取完整的瞬时游戏观测快照
    static bool build(Observation& out_obs);

    // 子系统只读提取接口
    static bool read_global(GlobalState& out_global);
    static bool read_cards(std::vector<CardState>& out_cards);
    static bool read_plants(std::vector<PlantState>& out_plants);
    static bool read_zombies(std::vector<ZombieState>& out_zombies);
    static bool read_dropped_cards(std::vector<DroppedCardState>& out_dropped_cards);
    static bool read_grid_items(std::vector<GridItemState>& out_grid_items);
    static bool read_ice_trails(std::vector<IceTrailState>& out_ice_trails);
    static bool read_mowers(std::vector<MowerState>& out_mowers);
    static bool read_grid(std::vector<GridSquareState>& out_grid);
};

}  // namespace pvz::observation
