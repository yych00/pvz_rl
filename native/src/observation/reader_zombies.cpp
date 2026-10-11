#include "reader_internal.h"
#include "zombie_decoder.h"
#include "boss_projectile_snapshot.h"

#include <cstdint>
#include <vector>

namespace pvz::observation {

bool read_zombies_nolock(PvzRuntime::VersionedPvZ& pvz, uintptr_t board, int grid_rows, std::vector<ZombieState>& out_zombies,
                         std::vector<ZombieState>* health,
                         std::vector<BossProjectileState>* boss_projectiles) {
    out_zombies.clear();
    if (health) health->clear();
    if (boss_projectiles) boss_projectiles->clear();
    if (!board || grid_rows <= 0) return false;
    // Beta 版在该结构中插入了额外字段，不能把零售版偏移当成事实读取。
    if (pvz.data().zombie_dead != 0xEC || pvz.data().zombie_status != 0x28) return false;
    const std::size_t item_size = PvzRuntime::zombie_struct_size(pvz);
    if (item_size != 0x15C && item_size != 0x168) return false;

    // 目标游戏始终是 32 位进程，即使当前 DLL 本身是 64 位。
    std::uint32_t zombie_arr = 0;
    std::int32_t zombie_max = 0;
    if (!pvz.read_remote_bytes(board + pvz.data().zombie, &zombie_arr, sizeof(zombie_arr)) ||
        !pvz.read_remote_bytes(board + pvz.data().zombie_count_max, &zombie_max, sizeof(zombie_max))) {
        return false;
    }
    if (zombie_max < 0 || zombie_max > 1024) return false;
    if (zombie_max == 0) return true;
    if (!zombie_arr) return false;

    // 一次带检查的大块读取替代每只僵尸的多次跨进程读取。
    // 读取失败时直接拒绝整份观测，避免发布全零字段或不完整僵尸列表。
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(zombie_max) * item_size);
    if (!pvz.read_remote_bytes(zombie_arr, bytes.data(), bytes.size())) return false;
    for (int slot = 0; slot < zombie_max; ++slot) {
        ZombieState zombie;
        if (decode_zombie(bytes.data() + static_cast<std::size_t>(slot) * item_size,
                          item_size, slot, grid_rows, zombie, true)) {
            if (health) health->push_back(zombie);
            if (!zombie.is_dead_or_dying) out_zombies.push_back(zombie);
            if (boss_projectiles && zombie.type == 25) {
                // Only this executable/layout has been verified for the ball
                // path. Never report an invented empty collection on a Boss
                // in an unsupported executable.
                if (pvz.version_id() != PVZ_1_2_0_1065_EN) return false;
                const auto lawn = pvz.read_memory_checked<std::uintptr_t>({pvz.data().lawn});
                if (!read_boss_projectile_snapshot(
                        [&](std::uintptr_t address, void* out, std::size_t length) {
                            return pvz.read_remote_bytes(address, out, length);
                        }, lawn + pvz.data().anim,
                        bytes.data() + static_cast<std::size_t>(slot) * item_size,
                        item_size, grid_rows, zombie.id, *boss_projectiles)) return false;
            }
        }
    }
    return true;
}

}  // 命名空间 pvz::observation
