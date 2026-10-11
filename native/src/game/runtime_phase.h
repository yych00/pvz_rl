#pragma once

#include "pvz/types.h"
#include "pvz/runtime.h"

#include <cstdint>

namespace PvzRuntime { class VersionedPvZ; }

namespace pvz::game {

// 高层运行阶段由原始 LawnApp::mGameScene 与 CutScene/Board 状态共同推导。
// 它与原始场景值分开：场景 2 同时包含开场/对话和交互式选卡，
// 冒险模式胜利也可能先通过 Board::mLevelComplete 出现，而不依赖奖励场景。
struct RuntimeState {
    bool process_available = false;
    std::int32_t raw_scene = -1;
    RuntimePhase phase = RuntimePhase::Offline;
    GameStatus status = GameStatus::Unknown;
    bool board_present = false;
    bool cutscene_present = false;
    bool seed_choosing = false;
    bool dave_talking = false;
    bool level_complete = false;
    std::int32_t cutscene_time = 0;
    std::int32_t dave_message = -1;
    std::int32_t game_mode = -1;
    std::int32_t level_id = 0;

    // PlayerInfo::mFinishedAdventure：0 表示尚未首次通关冒险，1 表示第二轮，
    // 更大的值表示后续循环并沿用通关后的进度规则；-1 表示当前无法读取。
    std::int32_t finished_adventure = -1;
    std::uintptr_t board_address = 0;
    std::int64_t game_clock = 0;
    bool game_clock_available = false;
};

class RuntimePhaseReader {
public:
    static RuntimeState read();
    // Caller owns ProcessManager::mutex. All required reads are checked; a
    // vanished/replaced Board cannot become a zero-valued successful state.
    static RuntimeState read_nolock(PvzRuntime::VersionedPvZ& pvz);

    // 纯状态分类函数。game_mode==0 表示冒险模式；只有冒险模式会把
    // Board::mLevelComplete 映射成终局阶段，因为挑战/生存模式可能在中间阶段复用该标志。
    static RuntimePhase classify(
        std::int32_t raw_scene,
        bool seed_choosing,
        bool dave_talking,
        bool level_complete = false,
        std::int32_t game_mode = -1);
    static GameStatus to_game_status(RuntimePhase phase);
    static const char* phase_name(RuntimePhase phase);
    static const char* raw_scene_name(std::int32_t raw_scene);
};

}  // 命名空间 pvz::game
