#pragma once

#include "pvz/types.h"

#include <memory>

namespace pvz {

class PvzEnv {
public:
    PvzEnv();
    ~PvzEnv();

    PvzEnv(const PvzEnv&) = delete;
    PvzEnv& operator=(const PvzEnv&) = delete;
    PvzEnv(PvzEnv&&) noexcept;
    PvzEnv& operator=(PvzEnv&&) noexcept;

    bool attach();
    bool is_game_on() const;
    GameStatus status() const;

    // 权威回合生命周期。reset 负责进入关卡，也可以停在交互式选卡界面，
    // 让同一会话继续完成选卡后再进入战斗。
    ResetResult reset(const ResetConfig& config);
    ResetResult wait_for_battle(std::int32_t timeout_ms, bool deterministic_pause);

    // 权威战斗状态转移。正式 Gym 调用方必须直接使用 StepResult，
    // 不在 Python 侧重新拼接动作、推进和观测调用。
    StepResult step(const Action& action, std::int32_t ticks);

    // 连续时间状态转移：不暂停或注入原版游戏循环。
    // 动作到达时立即执行，之后只等待到距离上次返回观测至少经过 target_ticks。
    // 如果策略计算延迟已经超过该间隔则立即返回；事件也从上次返回观测开始统计，
    // 因此智能体和模型的计算延迟会真实保留在状态转移中。
    StepResult step_continuous(const Action& action, std::int32_t target_ticks);

    Observation get_observation() const;
    std::vector<bool> get_action_mask() const;
    ActionResult execute_action(const Action& action);
    bool can_execute(const Action& action, ActionError* out_error = nullptr) const;

    bool pause();
    bool resume();
    bool is_paused() const;
    bool advance_ticks(std::int32_t ticks);
    std::int64_t get_game_clock() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // 命名空间 pvz
