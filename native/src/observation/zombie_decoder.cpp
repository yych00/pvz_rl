#include "zombie_decoder.h"

#include <cstring>

namespace pvz::observation {
namespace {

// 字段偏移已对照 Zombie.h 和本机 1.2.0.1065 可执行文件核实。
// 使用 memcpy 避免依赖宿主机对齐、指针宽度和 bool 表示方式。
template <typename T>
T read(const std::uint8_t* bytes, std::size_t offset) {
    T value;
    std::memcpy(&value, bytes + offset, sizeof(value));
    return value;
}

constexpr int kBungee = 20;
constexpr int kBoss = 25;
constexpr int kDying = 1;
constexpr int kMowered = 3;
constexpr int kBungeeDiving = 4;
constexpr int kBungeeRising = 8;
constexpr int kBossSpawning = 80;
constexpr int kBossStomping = 81;
constexpr int kBossBungeesEnter = 82;
constexpr int kBossBungeesLeave = 84;
constexpr int kBossDropRV = 85;

}  // 匿名命名空间

bool decode_zombie(const std::uint8_t* bytes, std::size_t size,
                   int slot, int grid_rows, ZombieState& out, bool include_dead) {
    if (!bytes || (size != 0x15C && size != 0x168) ||
        slot < 0 || slot > 0xFFFF || grid_rows <= 0) return false;

    const auto id = read<std::uint32_t>(bytes, size - sizeof(std::uint32_t));
    if ((id & 0xFFFF0000u) == 0 || (id & 0xFFFFu) != static_cast<unsigned>(slot)) {
        return false;  // 空闲链表项，或来自不同布局/快照的槽位。
    }
    const bool dead = read<std::uint8_t>(bytes, 0xEC) != 0;
    if (dead && !include_dead) return false;

    ZombieState z;
    z.id = read<std::int32_t>(bytes, size - sizeof(std::uint32_t));
    z.type = read<std::int32_t>(bytes, 0x24);
    z.phase = read<std::int32_t>(bytes, 0x28);
    // 存活判断对齐 Zombie::IsDeadOrDying，不根据剩余血量自行猜测。
    z.is_dead_or_dying = dead || (z.phase >= kDying && z.phase <= kMowered);
    if (z.type < 0 || (!include_dead && z.is_dead_or_dying)) return false;
    z.row = read<std::int32_t>(bytes, 0x1C);
    if (z.type != kBoss && (z.row < 0 || z.row >= grid_rows)) return false;

    z.phase_counter = read<std::int32_t>(bytes, 0x68);
    z.x = read<float>(bytes, 0x2C);
    z.y = read<float>(bytes, 0x30);
    z.vel_x = read<float>(bytes, 0x34);
    z.altitude = read<float>(bytes, 0x84);
    z.height = read<std::int32_t>(bytes, 0x64);
    z.body_hp = read<std::int32_t>(bytes, 0xC8);
    z.max_hp = read<std::int32_t>(bytes, 0xCC);
    z.helm_hp = read<std::int32_t>(bytes, 0xD0);
    z.shield_hp = read<std::int32_t>(bytes, 0xDC);
    z.flying_hp = read<std::int32_t>(bytes, 0xE4);
    z.helm_max_hp = read<std::int32_t>(bytes, 0xD4);
    z.shield_max_hp = read<std::int32_t>(bytes, 0xE0);
    z.flying_max_hp = read<std::int32_t>(bytes, 0xE8);
    z.from_wave = read<std::int32_t>(bytes, 0x6C);
    z.chilled_counter = read<std::int32_t>(bytes, 0xAC);
    z.buttered_counter = read<std::int32_t>(bytes, 0xB0);
    z.ice_trap_counter = read<std::int32_t>(bytes, 0xB4);
    z.is_eating = read<std::uint8_t>(bytes, 0x51) != 0;
    z.is_mind_controlled = read<std::uint8_t>(bytes, 0xB8) != 0;

    const auto valid_row = [grid_rows](int row) { return row >= 0 && row < grid_rows; };
    const auto valid_col = [](int col) { return col >= 0 && col < 9; };
    if (z.type == kBungee && z.phase >= kBungeeDiving && z.phase <= kBungeeRising) {
        const int col = read<std::int32_t>(bytes, 0x80);
        if (valid_row(z.row) && valid_col(col)) {
            // 蹦极选目标时调用 SetRow，不会写 mTargetRow，因此目标行直接取当前行。
            z.target_row = z.row;
            z.target_col = col;
        }
    } else if (z.type == kBoss) {
        // 僵王不同攻击之间会复用目标字段且不会主动清空，
        // 因此必须结合当前 phase 判断，不能把上一次攻击坐标当成当前目标。
        if (z.phase == kBossSpawning || z.phase == kBossStomping || z.phase == kBossDropRV) {
            const int row = read<std::int32_t>(bytes, 0x130);
            if (valid_row(row)) z.target_row = row;
        }
        if ((z.phase >= kBossBungeesEnter && z.phase <= kBossBungeesLeave) || z.phase == kBossDropRV) {
            const int col = read<std::int32_t>(bytes, 0x80);
            if (valid_col(col)) z.target_col = col;
        }
    }
    out = z;
    return true;
}

}  // 命名空间 pvz::observation
