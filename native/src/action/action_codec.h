#pragma once

#include "pvz/types.h"

namespace pvz::action {

// 固定战斗动作空间，各关卡维度保持不变：
//   0                 ：空操作
//   1..540            ：种植（卡槽、行、列），10×6×9
//   541..594          ：普通铲除（行、列），6×9
//   595..648          ：铲除南瓜（行、列），6×9
//   649..702          ：玉米炮发射（目标行、目标列），自动选择最早就绪炮
//   703..756          ：格子交互（行、列）；无原生入口时掩码关闭
//
// 普通铲除遵循原版铲除顺序：优先铲主植物；没有主植物时才铲睡莲或花盆。
// 南瓜模式表示直接铲除南瓜壳。当前局面中无效的行、卡槽或目标由原生动作掩码屏蔽。
class ActionCodec {
public:
    static constexpr int kMaxSlots = 10;
    static constexpr int kMaxRows = 6;
    static constexpr int kCols = 9;
    static constexpr int kShovelTargetCount = 2;

    static constexpr int kNoOpIndex = 0;
    static constexpr int kPlantBase = 1;
    static constexpr int kPlantCount = kMaxSlots * kMaxRows * kCols;
    static constexpr int kShovelBase = kPlantBase + kPlantCount;
    static constexpr int kShovelCells = kMaxRows * kCols;
    static constexpr int kShovelCount = kShovelTargetCount * kShovelCells;
    static constexpr int kCobBase = kShovelBase + kShovelCount;
    static constexpr int kCobTargetCells = kMaxRows * kCols;
    static constexpr int kCobCount = kCobTargetCells;
    static constexpr int kInteractBase = kCobBase + kCobCount;
    static constexpr int kInteractCount = kMaxRows * kCols;
    static constexpr int kActionSpaceSize = kInteractBase + kInteractCount; // 共 757 个动作。

    static bool decode(int index, Action& out_action);
    static int encode(const Action& action);
};

}  // 命名空间 pvz::action
