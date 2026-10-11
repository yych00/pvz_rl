#include "shovel_calls.h"

#include "../../process/process_manager.h"

#include <algorithm>
#include <cstdint>

namespace pvz::game::original {
namespace {
constexpr uintptr_t kPlantStructSize = 0x14C;

OriginalShovelTargetResult result(ActionError error, int id = -1) {
    return {true, error == ActionError::None, error, id};
}

void delete_plant(PvzRuntime::VersionedPvZ& pvz, uintptr_t plant) {
    if (!plant) return;
    pvz.asm_init();
#ifdef _PVZ_BETA_LEAK_SUPPORT
    if (pvz.isBETA()) {
        pvz.asm_mov_exx(Pt::Reg::ECX, static_cast<unsigned int>(plant));
    } else
#endif
    {
        pvz.asm_push_dword(static_cast<unsigned int>(plant));
    }
    pvz.asm_call(pvz.data().call_delete_plant);
    pvz.asm_ret();
    pvz.asm_code_inject();
}

OriginalShovelTargetResult resolve_unlocked(PvzRuntime::VersionedPvZ& pvz,
                                            int row, int col,
                                            ShovelTarget target_kind) {
    if (!pvz.GameOn()) return result(ActionError::GameNotFound);
    if (pvz.GameUI() != 3) return result(ActionError::GameNotInBattleState);
    const int target_value = static_cast<int>(target_kind);
    if (target_value < static_cast<int>(ShovelTarget::Normal) ||
        target_value > static_cast<int>(ShovelTarget::Pumpkin)) {
        return result(ActionError::InvalidArgument);
    }

    int rows = pvz.GetRowCount();
    if (rows <= 0) rows = 5;
    if (row < 0 || row >= rows || col < 0 || col >= 9) {
        return result(ActionError::InvalidGridCoordinates);
    }

    const uintptr_t board = pvz.ReadMemory<uintptr_t>({pvz.data().lawn, pvz.data().board});
    if (!board) return result(ActionError::GameNotInBattleState);
    const uintptr_t plants = pvz.ReadMemory<uintptr_t>({board + pvz.data().plant});
    int plant_max = pvz.ReadMemory<int>({board + pvz.data().plant_count_max});
    plant_max = std::max(0, std::min(1024, plant_max));

    int pumpkin = -1;
    int main_plant = -1;
    int carrier = -1;
    for (int i = 0; plants && i < plant_max; ++i) {
        const uintptr_t p = plants + static_cast<uintptr_t>(i) * kPlantStructSize;
        if (pvz.ReadMemory<bool>({p + pvz.data().plant_dead})) continue;
        if (pvz.ReadMemory<int>({p + pvz.data().plant_row}) != row ||
            pvz.ReadMemory<int>({p + pvz.data().plant_col}) != col) continue;
        const int type = pvz.ReadMemory<int>({p + pvz.data().plant_type});
        if (type == 30) pumpkin = i;
        else if (type == 16 || type == 33) carrier = i;
        else main_plant = i;
    }

    int target = -1;
    if (target_kind == ShovelTarget::Pumpkin) {
        target = pumpkin;
    } else {
        // Mirrors TOPPLANT_DIGGING_ORDER: main plant first, then under-plant.
        target = main_plant >= 0 ? main_plant : carrier;
    }

    if (target < 0) return result(ActionError::NoPlantToShovel);
    return result(ActionError::None, target);
}
}  // namespace

OriginalShovelTargetResult ShovelCalls::resolve_target(int row, int col, ShovelTarget target) {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return result(ActionError::GameNotFound);
    return resolve_unlocked(pvz, row, col, target);
}

bool ShovelCalls::build_flat_action_mask(std::vector<bool>& out_mask) {
    constexpr int kMaxRows = 6;
    constexpr int kCols = 9;
    constexpr int kCells = kMaxRows * kCols;
    out_mask.assign(2 * kCells, false);

    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return false;
    if (pvz.GameUI() != 3) return true;

    int rows = pvz.GetRowCount();
    if (rows <= 0) rows = 5;
    rows = std::min(rows, kMaxRows);

    const uintptr_t board = pvz.ReadMemory<uintptr_t>({pvz.data().lawn, pvz.data().board});
    if (!board) return false;
    const uintptr_t plants = pvz.ReadMemory<uintptr_t>({board + pvz.data().plant});
    int plant_max = pvz.ReadMemory<int>({board + pvz.data().plant_count_max});
    plant_max = std::max(0, std::min(1024, plant_max));

    for (int i = 0; plants && i < plant_max; ++i) {
        const uintptr_t p = plants + static_cast<uintptr_t>(i) * kPlantStructSize;
        // Match resolve_unlocked's liveness and raw-type checks exactly.
        if (pvz.ReadMemory<bool>({p + pvz.data().plant_dead})) continue;
        const int row = pvz.ReadMemory<int>({p + pvz.data().plant_row});
        if (row < 0 || row >= rows) continue;
        const int col = pvz.ReadMemory<int>({p + pvz.data().plant_col});
        if (col < 0 || col >= kCols) continue;
        const int type = pvz.ReadMemory<int>({p + pvz.data().plant_type});
        const int cell = row * kCols + col;
        // Normal shoveling accepts a main plant or carrier, never just a shell.
        // Target priority remains in resolve_target when executing the action.
        out_mask[static_cast<std::size_t>((type == 30 ? kCells : 0) + cell)] = true;
    }
    return true;
}

OriginalShovelTargetResult ShovelCalls::commit_target(int target_plant_id) {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return result(ActionError::GameNotFound);
    if (pvz.GameUI() != 3) return result(ActionError::GameNotInBattleState);

    const uintptr_t board = pvz.ReadMemory<uintptr_t>({pvz.data().lawn, pvz.data().board});
    if (!board) return result(ActionError::GameNotInBattleState);
    const uintptr_t plants = pvz.ReadMemory<uintptr_t>({board + pvz.data().plant});
    int plant_max = pvz.ReadMemory<int>({board + pvz.data().plant_count_max});
    plant_max = std::max(0, std::min(1024, plant_max));
    if (!plants || target_plant_id < 0 || target_plant_id >= plant_max) {
        return result(ActionError::NoPlantToShovel);
    }

    const uintptr_t plant = plants + static_cast<uintptr_t>(target_plant_id) * kPlantStructSize;
    if (pvz.ReadMemory<bool>({plant + pvz.data().plant_dead})) {
        return result(ActionError::NoPlantToShovel);
    }
    delete_plant(pvz, plant);
    return result(ActionError::None, target_plant_id);
}

}  // namespace pvz::game::original
