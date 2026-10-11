#pragma once

#include "runtime/pvz_runtime_layout.h"

#include <cstdint>
#include <string>

namespace pvz::runtime {

// 依据原版 LawnApp::IsScaryPotterStage (1.2.0.1065 地址 0x00453A30)
// 冒险模式 4-5 或解谜模式 51~60 (SCARY_POTTER)
inline bool is_scary_potter_stage_nolock(
    PvzRuntime::VersionedPvZ& pvz, uintptr_t board) {
    if (!board) return false;
    const int mode = pvz.read_memory_checked<int>({pvz.data().lawn, pvz.data().game_mode});
    if (mode >= 51 && mode <= 60) return true;
    if (mode == 0 && pvz.read_memory_checked<int>({board + pvz.data().adventure_level}) == 35) return true;
    return false;
}

// 依据原版 Whack a Zombie 规则 (冒险 2-5 或独立模式 30)
inline bool is_whack_a_zombie_stage_nolock(
    PvzRuntime::VersionedPvZ& pvz, uintptr_t board) {
    if (!board) return false;
    const int mode = pvz.read_memory_checked<int>({pvz.data().lawn, pvz.data().game_mode});
    if (mode == 30) return true;
    if (mode == 0 && pvz.read_memory_checked<int>({board + pvz.data().adventure_level}) == 15) return true;
    return false;
}

inline int seed_bank_slot_count_nolock(
    PvzRuntime::VersionedPvZ& pvz, uintptr_t board) {
    if (!board) return 0;
    // 砸罐子关卡 (4-5 及解谜 51~60) 无有效卡槽，卡片全部来自掉落物 (UsableSeedPacket)
    if (is_scary_potter_stage_nolock(pvz, board)) {
        return 0;
    }
    const uintptr_t seed_bank = pvz.read_memory_checked<uintptr_t>({board + pvz.data().slot});
    if (!seed_bank) return 0;
    const int count = pvz.read_memory_checked<int>({seed_bank + pvz.data().slot_count});
    if (count < 0 || count > 10)
        throw PvzRuntime::NativeOperationError(ActionError::Unsupported, "invalid seed bank slot count");
    return count;
}

// 依据原版 Board::HasConveyorBelt (1.2.0.1065 地址 0x0041BE80) 反汇编逆向提取的事实规则。
// 纯内存读取，0 开销，与原版游戏行为严格一致。
inline bool has_conveyor_seed_bank_nolock(
    PvzRuntime::VersionedPvZ& pvz, uintptr_t board) {
    if (!board) return false;
    const int mode = pvz.read_memory_checked<int>({pvz.data().lawn, pvz.data().game_mode});
    if (mode == 0) {
        // 冒险模式传送带关卡：1-5, 1-10, 2-10, 3-5, 3-10, 4-10(雨夜), 5-5(蹦极), 5-10(僵博)
        // 注意：4-5(35) 是砸罐子(Vasebreaker)，原版 0x453A30 明确判定为 Scary Potter，非传送带！
        switch (pvz.read_memory_checked<int>({board + pvz.data().adventure_level})) {
        case 5:
        case 10:
        case 20:
        case 25:
        case 30:
        case 40:
        case 45:
        case 50:
            return true;
        default:
            return false;
        }
    }

    // 挑战与小游戏模式传送带
    switch (mode) {
    case 17: // Wall-nut Bowling
    case 21: // Invisi-ghoul
    case 25: // Little Trouble
    case 26: // Portal Combat
    case 27: // Column Like You See 'Em
    case 33: // Wall-nut Bowling 2
    case 35: // Dr. Zomboss's Revenge
    case 46: // Shovel
    case 47: // Bungee Blitz
    case 48: // Stormy Night
        return true;
    default:
        return false;
    }
}

// 判定当前关卡的卡牌供给系统类型：
// "conveyor"：传送带
// "dropped"：纯掉落物（无卡槽，如 19 It's Raining Seeds, 4-5 砸罐子）
// "seed_bank"：常规选卡卡槽
inline const char* card_system_type_nolock(
    PvzRuntime::VersionedPvZ& pvz, uintptr_t board) {
    if (!board) return "unknown";
    if (has_conveyor_seed_bank_nolock(pvz, board)) {
        return "conveyor";
    }
    if (is_scary_potter_stage_nolock(pvz, board)) {
        return "dropped";
    }
    const int slot_count = seed_bank_slot_count_nolock(pvz, board);
    if (slot_count == 0) {
        return "dropped";
    }
    return "seed_bank";
}

// 获得可读的游戏模式标准英文名称
inline std::string game_mode_name_nolock(int mode, int level_id = 0) {
    if (mode == 0) {
        if (level_id <= 0) return "Adventure";
        const int area = (level_id - 1) / 10 + 1;
        const int sub = (level_id - 1) % 10 + 1;
        return "Adventure " + std::to_string(area) + "-" + std::to_string(sub);
    }

    switch (mode) {
    case 1:  return "Survival: Day";
    case 2:  return "Survival: Night";
    case 3:  return "Survival: Pool";
    case 4:  return "Survival: Fog";
    case 5:  return "Survival: Roof";
    case 6:  return "Survival: Day (Hard)";
    case 7:  return "Survival: Night (Hard)";
    case 8:  return "Survival: Pool (Hard)";
    case 9:  return "Survival: Fog (Hard)";
    case 10: return "Survival: Roof (Hard)";
    case 11: return "Survival: Day (Endless)";
    case 12: return "Survival: Night (Endless)";
    case 13: return "Survival: Pool (Endless)";
    case 14: return "Survival: Fog (Endless)";
    case 15: return "Survival: Roof (Endless)";
    case 17: return "Wall-nut Bowling";
    case 18: return "Slot Machine";
    case 19: return "It's Raining Seeds";
    case 20: return "Beghouled";
    case 21: return "Invisighoul";
    case 22: return "Seeing Stars";
    case 23: return "Zombiquarium";
    case 24: return "Beghouled Twist";
    case 25: return "Little Trouble";
    case 26: return "Portal Combat";
    case 27: return "Column Like You See 'Em";
    case 28: return "Bobsled Bonanza";
    case 29: return "Zombie Nimble Zombie Quick";
    case 30: return "Whack a Zombie";
    case 31: return "Last Stand";
    case 32: return "Zombotany";
    case 33: return "Wall-nut Bowling 2";
    case 35: return "Dr. Zomboss's Revenge";
    case 46: return "Art Challenge Wall-nut";
    case 47: return "Bungee Blitz";
    case 48: return "Stormy Night";
    case 49: return "Sunny Day";
    case 50: return "Art Challenge Sunflower";
    default: return "Custom Mode " + std::to_string(mode);
    }
}

inline bool conveyor_packet_visible_nolock(
    PvzRuntime::VersionedPvZ& pvz, uintptr_t seed_bank, int slot) {
    constexpr int kConveyorClipRight = 591;
    constexpr int kFirstPacketX = 91;
    constexpr int kPacketSpacing = 50;
    const int offset_x = pvz.read_memory_checked<int>({
        seed_bank + static_cast<uintptr_t>(slot) * 0x50 + 0x58});
    return kFirstPacketX + kPacketSpacing * slot + offset_x < kConveyorClipRight;
}

} // namespace pvz::runtime
