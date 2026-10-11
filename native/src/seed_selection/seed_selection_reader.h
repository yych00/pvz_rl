#pragma once

#include "pvz/seed_selection.h"
#include <cstdint>

namespace PvzRuntime {
class VersionedPvZ;
}

namespace pvz::seed_selection {

// 检查指定植物类型在当前玩家进度下是否真正解锁/可用 (调用原版 LawnApp::SeedTypeAvailable)
bool seed_type_available(
    PvzRuntime::VersionedPvZ& pvz,
    uintptr_t lawn,
    int plant_type);

// 批量查询植物解锁状态 (单次汇编注入，极速批量获取)
void query_all_seed_types_available(
    PvzRuntime::VersionedPvZ& pvz,
    uintptr_t lawn,
    bool* out_available,
    int count);

class SeedSelectionReader {
public:
    static bool read(SeedSelectionState& out);
};

}  // namespace pvz::seed_selection
