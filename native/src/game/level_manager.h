#pragma once

#include "pvz/types.h"

#include <cstdint>

namespace pvz::game {

// 关卡生命周期与 UI 驱动
class LevelManager {
public:
    // 选择冒险模式关卡 (level_id: 1~50)
    static int select_level(int level_id);

    // 重新开始当前关卡
    static int restart_level();

    // 等待游戏进入就绪战斗状态 (消除开场动画、UI==3 且可行动)
    static bool wait_until_ready(int timeout_ms = 10000);

    // 获取当前关卡 ID (1~50)
    static int current_level();

    // 获取结构化当前游戏状态 (Menu / ChoosingSeeds / Playing / Won / Lost)
    static GameStatus get_game_status();

    // 判断当前是否胜利或失败
    static void check_game_outcome(bool& out_won, bool& out_lost);
};

}  // namespace pvz::game
