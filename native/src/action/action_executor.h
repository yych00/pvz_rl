#pragma once

#include "pvz/types.h"

namespace pvz::action {

// 真正修改游戏前会再次解析合法性；动作掩码只是当前时刻的快照。
class ActionExecutor {
public:
    static ActionResult execute(const Action& action);

    static ActionResult plant(int slot, int row, int col);
    static ActionResult shovel(int row, int col, ShovelTarget target);
    static ActionResult cob_fire(int target_row, int target_col);
};

}  // 命名空间 pvz::action
