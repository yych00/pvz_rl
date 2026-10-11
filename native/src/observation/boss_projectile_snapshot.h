#pragma once

#include "pvz/types.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace pvz::observation {

// Verified in the user's 1.2.0.1065 executable, UpdateBossFireball at
// 0x535FB0 and BossHeadSpitContact at 0x535DC0. See the fact audit in docs.
// All pointers and IDs below describe the 32-bit target, including x64 DLLs.
template<class T> T boss_field(const std::uint8_t* bytes, std::size_t offset) {
    T value;
    std::memcpy(&value, bytes + offset, sizeof(value));
    return value;
}

template<class Read>
bool read_boss_projectile_snapshot(Read&& read, std::uintptr_t effect_pointer_address,
                                  const std::uint8_t* zombie_bytes, std::size_t size,
                                  int grid_rows, int owner_id,
                                  std::vector<BossProjectileState>& output) {
    if (!zombie_bytes || size != 0x15C || grid_rows <= 0) return false;
    const auto id = boss_field<std::uint32_t>(zombie_bytes, 0x140);
    if (id == 0) return true; // Stale row/type fields alone are not an entity.
    const int row = boss_field<std::int32_t>(zombie_bytes, 0x148);
    const bool fire = boss_field<std::uint8_t>(zombie_bytes, 0x14C) != 0;
    if (row < 0 || row >= grid_rows) return false;

    std::uint32_t effects = 0, holder = 0, items = 0;
    std::int32_t capacity = 0;
    if (!read(effect_pointer_address, &effects, 4) || !effects ||
        !read(static_cast<std::uintptr_t>(effects) + 0x08, &holder, 4) || !holder ||
        !read(holder, &items, 4) || !items ||
        !read(static_cast<std::uintptr_t>(holder) + 0x08, &capacity, 4)) return false;
    if (capacity <= 0 || capacity > 4096) return false;
    const auto slot = id & 0xFFFFu;
    // ReanimationTryToGet rejects out-of-range and recycled IDs as absent.
    if (slot >= static_cast<unsigned>(capacity)) return true;
    std::uint8_t reanim[0xA0];
    if (!read(static_cast<std::uintptr_t>(items) + slot * sizeof(reanim),
              reanim, sizeof(reanim))) return false;
    if (boss_field<std::uint32_t>(reanim, 0x9C) != id || reanim[0x14]) return true;
    if (boss_field<std::int32_t>(reanim, 0x00) != (fire ? 0x54 : 0x53)) return false;

    BossProjectileState entity;
    entity.id = id;
    entity.owner_id = owner_id;
    entity.type = fire ? 0 : 1;
    entity.row = row;
    entity.x = boss_field<float>(reanim, 0x2C);
    entity.y = boss_field<float>(reanim, 0x38);
    if (!std::isfinite(entity.x) || !std::isfinite(entity.y)) return false;
    entity.valid = true;
    output.push_back(entity);
    return true;
}

} // namespace pvz::observation
