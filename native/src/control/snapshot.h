#pragma once

#include "pvz/types.h"

#include <cstdint>
#include <vector>

namespace pvz::control {

// 游戏原生局面快照数据结构 (用于失败复训、debug 与 curriculum)
struct GameSnapshot {
    std::int64_t game_clock = 0;
    std::int32_t level_id = 0;
    std::vector<std::uint8_t> board_memory_dump; // 游戏主战场内存数据镜像
    Observation cached_observation;
};

// 游戏状态存档与回档控制
class SnapshotManager {
public:
    // 保存当前游戏原生局面
    static bool save_state(GameSnapshot& out_snapshot);

    // 回档至目标快照 (恢复后必须强制刷新最新 Observation)
    static bool load_state(const GameSnapshot& snapshot, Observation& out_new_obs);
};

}  // namespace pvz::control
