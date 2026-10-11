#pragma once
#include "pvz/types.h"
#include <cstdint>
#include <cstring>
#include <vector>

namespace pvz::observation {
constexpr std::size_t kPlantStride = 0x14c;
template<class T> T plant_field(const unsigned char* slot, std::size_t offset) {
    T value{}; std::memcpy(&value, slot + offset, sizeof(value)); return value;
}
// Retail/GOTY layout; unsupported layouts must be rejected by the caller.
inline void decode_plants(const std::vector<unsigned char>& bytes, int rows,
                          std::vector<PlantState>& out) {
    out.clear();
    for (std::size_t i = 0; i < bytes.size() / kPlantStride; ++i) {
        const auto* p = bytes.data() + i * kPlantStride;
        const auto id = plant_field<std::uint32_t>(p, 0x148);
        if (!(id & 0xffff0000u) || (id & 0xffffu) != i || p[0x141] || p[0x142]) continue;
        PlantState s{};
        s.id = static_cast<std::int32_t>(id);
        s.row = plant_field<std::int32_t>(p, 0x1c);
        s.col = plant_field<std::int32_t>(p, 0x28);
        s.type = plant_field<std::int32_t>(p, 0x24);
        s.hp = plant_field<std::int32_t>(p, 0x40);
        s.max_hp = plant_field<std::int32_t>(p, 0x44);
        if (s.row < 0 || s.row >= rows || s.col < 0 || s.col >= 9 || s.type < 0 ||
            (s.hp <= 0 && s.max_hp > 0)) continue;
        s.layer = (s.type == 16 || s.type == 33) ? 0 : s.type == 30 ? 2 : 1;
        s.state = plant_field<std::int32_t>(p, 0x3c);
        s.is_sleeping = p[0x143] != 0;
        s.imitater_type = plant_field<std::int32_t>(p, 0x138);
        out.push_back(s);
    }
}
template<class Read> bool read_plant_snapshot(Read read, std::uintptr_t ptr_address,
        std::uintptr_t count_address, int rows, std::vector<PlantState>& out) {
    out.clear();
    std::uint32_t array = 0; std::int32_t count = 0;
    if (!read(ptr_address, &array, sizeof(array)) ||
        !read(count_address, &count, sizeof(count)) || count < 0 || count > 1024) return false;
    if (!count) return true;
    if (!array || std::uint64_t(array) + std::uint64_t(count) * kPlantStride > 0x100000000ull) return false;
    std::vector<unsigned char> bytes(std::size_t(count) * kPlantStride);
    if (!read(array, bytes.data(), bytes.size())) return false;
    decode_plants(bytes, rows, out);
    return true;
}
}
