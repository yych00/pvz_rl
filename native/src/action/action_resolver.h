#pragma once

#include "pvz/types.h"
#include "../game/rules/rule_types.h"

#include <variant>

namespace pvz::action {

struct NoOpResolution {};

struct PlantResolution {
    int slot = -1;
    int row = -1;
    int col = -1;
};

struct ShovelResolution {
    int row = -1;
    int col = -1;
    ShovelTarget target = ShovelTarget::Normal;
    int target_plant_id = -1;
};

struct CobFireResolution {
    int source_row = -1;
    int source_col = -1;
    int target_row = -1;
    int target_col = -1;
    int cannon_id = -1;
    int target_x = -1;
    int target_y = -1;
    std::uint32_t cannon_identity = 0;
};

struct VaseResolution {
    int row = -1;
    int col = -1;
    std::uint32_t identity = 0;
    uintptr_t board = 0;
};

using ResolutionPayload = std::variant<
    NoOpResolution,
    PlantResolution,
    ShovelResolution,
    CobFireResolution,
    VaseResolution>;

struct ActionResolution {
    bool legal = false;
    ActionError error = ActionError::Unsupported;
    game::rules::RuleSource source = game::rules::RuleSource::Unsupported;
    ResolutionPayload payload = NoOpResolution{};
};

// 验证、动作掩码和实际执行共用的唯一合法性解析入口。
class ActionResolver {
public:
    static ActionResolution resolve(const Action& action);
};

}  // 命名空间 pvz::action
