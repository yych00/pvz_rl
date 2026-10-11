#pragma once

#include "pvz/types.h"

namespace pvz::action {

// 兼容只关心“合法/非法”的调用方；权威游戏语义由 ActionResolver 和规则层统一提供。
class ActionValidator {
public:
    static bool can_execute(const Action& action, ActionError* out_error = nullptr);
};

}  // 命名空间 pvz::action
