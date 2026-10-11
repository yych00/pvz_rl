#pragma once

#include "pvz/types.h"

#include <cstddef>
#include <cstdint>

namespace pvz::observation {

// 解码零售版/年度版 DataArrayItem 快照，包含末尾的代数 ID。
// Beta 布局暂不支持，避免用未核实偏移发布错误事实。
bool decode_zombie(const std::uint8_t* bytes, std::size_t size,
                   int slot, int grid_rows, ZombieState& out,
                   bool include_dead = false);

}  // 命名空间 pvz::observation
