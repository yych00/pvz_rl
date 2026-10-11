#include "runtime/remote_allocation.h"
#include "seed_selection_reader.h"

#include "../process/process_manager.h"
#include "../process/runtime.h"
#include "runtime/pvz_runtime_layout.h"

#include <algorithm>
#include <set>
#include <vector>

namespace pvz::seed_selection {
namespace {

constexpr uintptr_t kCutsceneSeedChoosingOffset = 0x2C;

bool is_seed_choosing_nolock(PvzRuntime::VersionedPvZ& pvz, uintptr_t lawn) {
    // LawnApp::mGameScene==SCENE_LEVEL_INTRO is necessary but not sufficient:
    // SeedChooserScreen is created during the whole intro. The original
    // CutScene::StartSeedChooser()/EndSeedChooser() authority is mSeedChoosing.
    if (!pvz.GameOn() || !lawn || pvz.read_memory_checked<int>({pvz.data().lawn, pvz.data().game_ui}) != 2) return false;

    const uintptr_t board = pvz.read_memory_checked<uintptr_t>({lawn + pvz.data().board});
    if (!board) return false;

    const uintptr_t cutscene = pvz.read_memory_checked<uintptr_t>({board + pvz.data().cut_scene});
    if (!cutscene) return false;

    return pvz.read_memory_checked<bool>({cutscene + kCutsceneSeedChoosingOffset});
}

uint32_t compute_warnings(const std::vector<int>& selected, int current_level_num) {
    uint32_t flags = static_cast<uint32_t>(SeedWarning::None);
    std::set<int> sel(selected.begin(), selected.end());

    // 1. NoSunProducer: 向日葵(1), 阳光菇(9), 双胞向日葵(40)
    if (!sel.count(1) && !sel.count(9) && !sel.count(40)) {
        flags |= static_cast<uint32_t>(SeedWarning::NoSunProducer);
    }

    // 关卡判断：adventure level 1..50
    // 关卡 21..40 为泳池/浓雾 (Area 3, 4)
    if (current_level_num >= 21 && current_level_num <= 40) {
        // 荷叶(16)
        if (!sel.count(16)) {
            flags |= static_cast<uint32_t>(SeedWarning::NoPoolSupport);
        }
    }
    // 关卡 41..50 为屋顶 (Area 5)
    if (current_level_num >= 41 && current_level_num <= 50) {
        // 花盆(33)
        if (!sel.count(33)) {
            flags |= static_cast<uint32_t>(SeedWarning::NoFlowerPot);
        }
    }

    // UpgradeBaseMissing:
    // 双胞向日葵(40) 需要 向日葵(1)
    if (sel.count(40) && !sel.count(1)) flags |= static_cast<uint32_t>(SeedWarning::UpgradeBaseMissing);
    // 机枪射手(41) 需要 双发射手(7)
    if (sel.count(41) && !sel.count(7)) flags |= static_cast<uint32_t>(SeedWarning::UpgradeBaseMissing);
    // 冰西瓜(42) 需要 西瓜投手(34)
    if (sel.count(42) && !sel.count(34)) flags |= static_cast<uint32_t>(SeedWarning::UpgradeBaseMissing);
    // 地刺王(44) 需要 地刺(22)
    if (sel.count(44) && !sel.count(22)) flags |= static_cast<uint32_t>(SeedWarning::UpgradeBaseMissing);
    // 玉米加农炮(45) 需要 玉米投手(32)
    if (sel.count(45) && !sel.count(32)) flags |= static_cast<uint32_t>(SeedWarning::UpgradeBaseMissing);

    return flags;
}

}  // namespace

void query_all_seed_types_available(
    PvzRuntime::VersionedPvZ& pvz,
    uintptr_t lawn,
    bool* out_available,
    int count) {
    if (!out_available || count <= 0) return;
    for (int i = 0; i < count; ++i) {
        out_available[i] = false;
    }

    // Only call original SeedTypeAvailable while the chooser is genuinely
    // interactive. Raw scene 2 alone also covers Dave/intro animation.
    if (!is_seed_choosing_nolock(pvz, lawn)) return;

    const uintptr_t fn = PvzRuntime::seed_type_available_address(pvz);
    if (!fn) return;

    pvz.ensure_remote_idle();
    HANDLE hProc = pvz.process_handle();
    if (!hProc) return;

    LPVOID remote_out = VirtualAllocEx(
        hProc, nullptr, static_cast<SIZE_T>(count),
        MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!remote_out) throw PvzRuntime::NativeOperationError(pvz::ActionError::Unsupported, "seed availability allocation failed");
    PvzRuntime::RemoteAllocation allocation(hProc, remote_out);

    // LawnApp::SeedTypeAvailable calling convention:
    // EDI = LawnApp*, EBX = SeedType, return AL.
    pvz.asm_init();
    pvz.asm_add_byte(0x60); // pushad
    pvz.asm_mov_exx(Pt::Reg::EDI, static_cast<unsigned int>(lawn));
    pvz.asm_add_list(0x31, 0xf6); // xor esi, esi

    // loop_start:
    pvz.asm_add_list(0x89, 0xf3); // mov ebx, esi
    pvz.asm_call(static_cast<unsigned int>(fn));
    pvz.asm_mov_exx(Pt::Reg::EDX, static_cast<unsigned int>(reinterpret_cast<uintptr_t>(remote_out)));
    pvz.asm_add_list(0x88, 0x04, 0x32); // mov [edx + esi], al
    pvz.asm_add_byte(0x46); // inc esi
    pvz.asm_add_list(0x83, 0xfe, static_cast<unsigned char>(count)); // cmp esi, count
    pvz.asm_add_list(0x7c, static_cast<unsigned char>(-21)); // jl loop_start

    pvz.asm_add_byte(0x61); // popad
    pvz.asm_ret();
    pvz.asm_code_inject();

    std::vector<uint8_t> temp_buf(count, 0);
    if (!pvz.read_remote_bytes(
        reinterpret_cast<uintptr_t>(remote_out),
        temp_buf.data(), count)) {
        throw PvzRuntime::NativeOperationError(pvz::ActionError::Unsupported, "seed availability result read failed");
    }

    for (int i = 0; i < count; ++i) {
        out_available[i] = (temp_buf[i] != 0);
    }


}

bool seed_type_available(
    PvzRuntime::VersionedPvZ& pvz,
    uintptr_t lawn,
    int plant_type) {
    if (plant_type < 0 || plant_type >= kSeedTypeCapacity) {
        return false;
    }
    bool avail[kSeedTypeCapacity] = {false};
    query_all_seed_types_available(pvz, lawn, avail, kNumSeedsInChooser);
    if (plant_type < kNumSeedsInChooser) {
        return avail[plant_type];
    }
    return false;
}

bool SeedSelectionReader::read(SeedSelectionState& out) {
    out = SeedSelectionState{};

    if (!process::game_on()) {
        return false;
    }

    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn()) {
        return false;
    }

    const uintptr_t lawn = pvz.read_memory_checked<uintptr_t>({pvz.data().lawn});
    if (!lawn) {
        out.present = false;
        return true;
    }

    // Authoritative interactive chooser predicate:
    // SCENE_LEVEL_INTRO + CutScene::mSeedChoosing=true.
    // mSeedChooserScreen may already exist earlier in the intro, so pointer
    // presence alone is not enough to declare SeedSelection.
    if (!is_seed_choosing_nolock(pvz, lawn)) {
        out.present = false;
        return true;
    }

    // LawnApp + 0x774 = mSeedChooserScreen.
    const uintptr_t chooser = pvz.read_memory_checked<uintptr_t>({lawn + 0x774});
    if (!chooser) {
        out.present = false;
        return true;
    }
    out.present = true;

    out.selected_count = pvz.read_memory_checked<int>({chooser + 0xd24});
    int seeds_in_flight = pvz.read_memory_checked<int>({chooser + 0xd20});
    if (seeds_in_flight < 0) {
        seeds_in_flight = 0;
    }

    int required_count = 0;
    const uintptr_t board = pvz.read_memory_checked<uintptr_t>({chooser + 0xd14});
    if (board) {
        const uintptr_t seed_bank = pvz.read_memory_checked<uintptr_t>({board + 0x144});
        if (seed_bank) {
            required_count = pvz.read_memory_checked<int>({seed_bank + 0x24});
        }
    }
    if (required_count <= 0 || required_count > 10) {
        throw PvzRuntime::NativeOperationError(ActionError::Unsupported, "invalid chooser required seed count");
    }
    out.required_count = required_count;

    bool unlocked_arr[kNumSeedsInChooser] = {false};
    query_all_seed_types_available(pvz, lawn, unlocked_arr, kNumSeedsInChooser);

    std::vector<std::pair<int, int>> bank_seeds; // (index_in_bank, plant_type)
    for (int plant_type = 0; plant_type < kNumSeedsInChooser; ++plant_type) {
        const uintptr_t seed_addr = chooser + 0xa4 + static_cast<uintptr_t>(plant_type * 0x3c);
        const int internal_seed_type = pvz.read_memory_checked<int>({seed_addr + 0x20});
        const int seed_state = pvz.read_memory_checked<int>({seed_addr + 0x24});
        const int bank_idx = pvz.read_memory_checked<int>({seed_addr + 0x28});
        const int dave_picked = pvz.read_memory_checked<int>({seed_addr + 0x38});

        if (internal_seed_type != plant_type) {
            // Defensive sanity signal only; ABI action id stays the array index.
        }

        if (seed_state == 0 || seed_state == 2) {
            seeds_in_flight++;
        }

        const bool unlocked = unlocked_arr[plant_type];

        SeedOptionState opt;
        opt.plant_type = plant_type;
        opt.chooser_state = seed_state;
        opt.unlocked = unlocked;
        opt.selected = unlocked && (seed_state == 0 || seed_state == 1);
        opt.available = unlocked && (seed_state == 3) && (dave_picked == 0);
        opt.disabled = !unlocked || (seed_state == 4) || (dave_picked != 0);

        out.options.push_back(opt);

        if (opt.selected) {
            bank_seeds.push_back({bank_idx, plant_type});
            if (plant_type == kImitaterType) {
                out.imitater_target = pvz.read_memory_checked<int>({seed_addr + 0x34});
            }
        }
    }

    out.seeds_in_flight = (seeds_in_flight > 0);
    out.can_confirm = (!out.seeds_in_flight && out.selected_count == out.required_count);

    std::sort(bank_seeds.begin(), bank_seeds.end());
    out.selected_seeds.clear();
    for (const auto& p : bank_seeds) {
        out.selected_seeds.push_back(p.second);
    }

    int current_lvl = 0;
    if (board) {
        current_lvl = pvz.read_memory_checked<int>({board + pvz.data().adventure_level});
        // Board::mZombieAllowed, same versioned layout as reader_global.cpp.
        // A failed read must remain distinguishable from an empty roster.
        uint8_t allowed[33] = {};
        const uintptr_t version_offset = pvz.data().sun - 0x5560;
        if (pvz.read_remote_bytes(board + 0x54D4 + version_offset,
                                  allowed, sizeof(allowed)) &&
            std::all_of(std::begin(allowed), std::end(allowed),
                        [](uint8_t value) { return value <= 1; })) {
            out.zombie_roster_available = true;
            for (int zid = 0; zid < 33; ++zid) {
                if (allowed[zid]) out.zombie_types_in_level.push_back(zid);
            }
        }
    }
    out.warning_flags = compute_warnings(out.selected_seeds, current_lvl);

    return true;
}

}  // namespace pvz::seed_selection
