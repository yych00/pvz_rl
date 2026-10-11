#pragma once

#include "action_codec.h"

#include <vector>

namespace pvz::action {

class ActionMask {
public:
    static constexpr int kActionSpaceSize = ActionCodec::kActionSpaceSize;

    // 空操作和铲除走统一验证器；种植和玉米炮使用批量原版查询，
    // 避免为每个空间组合重复扫描远程内存。
    static void compute_mask(std::vector<bool>& out_mask);
};

}  // 命名空间 pvz::action
