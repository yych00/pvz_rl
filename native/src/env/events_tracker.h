#pragma once

#include "pvz/types.h"

#include <unordered_set>

namespace pvz::env {

// 单步区间事实追踪器，只记录事实，奖励策略留在 Python。
// 基于实体消失的统计会过滤玩家主动铲除、升级替换和一次性植物自消耗等情况，
// 避免把每次 ID 变化都误判为战损。
class EventsTracker {
public:
    EventsTracker() = default;

    void begin_step(const Observation& current_obs);
    StepEvents end_step(
        const Observation& new_obs,
        const Action& action,
        const ActionResult& action_result,
        GameStatus new_status);
    void reset();

private:
    Observation baseline_obs_;
    std::unordered_set<int> living_zombie_ids_;
    std::unordered_set<int> living_plant_ids_;
    int baseline_sun_ = 0;
    int baseline_wave_ = 0;
    bool has_baseline_ = false;
};

}  // 命名空间 pvz::env
