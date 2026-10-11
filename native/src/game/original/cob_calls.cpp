#include "cob_calls.h"

#include "../../process/process_manager.h"
#include "runtime/pvz_runtime_layout.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <tuple>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pvz::game::original {
namespace {
constexpr uintptr_t kPlantStructSize = 0x14C;
constexpr int kBoardWidth = 800;
constexpr int kBoardHeight = 600;
constexpr int kLawnXMin = 40;
constexpr int kLawnYMin = 80;
constexpr int kCellWidth = 80;
constexpr int kGrassCellHeight = 100;
constexpr int kPoolCellHeight = 85;
constexpr int kRoofCellHeight = 85;

struct ReadyCannon {
    int id;
    int row;
    int col;
    std::int64_t ready_since;
    std::uint32_t identity;
};

uintptr_t g_ready_board = 0;
std::uint32_t g_ready_pid = 0;
int g_last_clock = -1;
std::unordered_map<std::uint32_t, std::int64_t> g_ready_since;

// ProcessManager's mutex protects this history. A cannon first seen ready in
// one snapshot receives that snapshot's game tick; tied cannons use grid order.
std::vector<ReadyCannon> ready_cannons_unlocked(
    PvzRuntime::VersionedPvZ& pvz, uintptr_t board, int rows) {
    const std::uint32_t pid = pvz.process_id();
    const int clock = pvz.ReadMemory<int>({board + pvz.data().game_clock});
    if (g_ready_pid != pid || g_ready_board != board || clock < g_last_clock) {
        g_ready_since.clear();
        g_ready_pid = pid;
        g_ready_board = board;
    }
    g_last_clock = clock;
    const uintptr_t plants = pvz.ReadMemory<uintptr_t>({board + pvz.data().plant});
    int plant_max = pvz.ReadMemory<int>({board + pvz.data().plant_count_max});
    plant_max = std::max(0, std::min(1024, plant_max));
    std::unordered_set<std::uint32_t> still_ready;
    std::vector<ReadyCannon> ready;
    for (int i = 0; plants && i < plant_max; ++i) {
        const uintptr_t p = plants + static_cast<uintptr_t>(i) * kPlantStructSize;
        if (pvz.ReadMemory<bool>({p + pvz.data().plant_dead}) ||
            pvz.ReadMemory<int>({p + pvz.data().plant_type}) != 47 ||
            pvz.ReadMemory<int>({p + PvzRuntime::kPlantStateOffset}) !=
                PvzRuntime::kCobCannonReadyState) continue;
        const int row = pvz.ReadMemory<int>({p + pvz.data().plant_row});
        const int col = pvz.ReadMemory<int>({p + pvz.data().plant_col});
        if (row < 0 || row >= rows || col < 0 || col >= 8) continue;
        const std::uint32_t identity = pvz.ReadMemory<std::uint32_t>({
            p + kPlantStructSize - sizeof(std::uint32_t)});
        if ((identity & 0xffff0000u) == 0 ||
            (identity & 0xffffu) != static_cast<std::uint32_t>(i)) continue;
        still_ready.insert(identity);
        const auto it = g_ready_since.emplace(identity, clock).first;
        ready.push_back({i, row, col, it->second, identity});
    }
    for (auto it = g_ready_since.begin(); it != g_ready_since.end();) {
        if (!still_ready.count(it->first)) it = g_ready_since.erase(it);
        else ++it;
    }
    std::sort(ready.begin(), ready.end(), [](const auto& a, const auto& b) {
        return std::tie(a.ready_since, a.row, a.col, a.id) <
               std::tie(b.ready_since, b.row, b.col, b.id);
    });
    return ready;
}

OriginalCobCallResult result(ActionError error, int cannon_id = -1,
                             int target_x = -1, int target_y = -1,
                             std::uint32_t cannon_identity = 0) {
    return {true, error == ActionError::None, error, cannon_id, target_x, target_y,
            cannon_identity};
}

// CobCannonFire(x, y) consumes the exact board-local mouse target, not the
// Board::GridToPixelX/Y plant-placement anchor. Convert the discrete RL target
// cell to the center of the actual PixelToGridX/Y cell instead.
bool grid_to_target_center(PvzRuntime::VersionedPvZ& pvz, int row, int col,
                           int& out_x, int& out_y) {
    int rows = pvz.GetRowCount();
    if (rows <= 0) rows = 5;
    if (row < 0 || row >= rows || col < 0 || col >= 9) return false;

    // PixelToGridX: (x - LAWN_XMIN) / 80.
    out_x = kLawnXMin + col * kCellWidth + kCellWidth / 2;

    const int scene = pvz.GetScene();
    if (scene == 4 || scene == 5) { // roof
        // PixelToGridY subtracts (4 - col) * 20 before dividing by 85 for
        // the sloped half of the roof. Reconstruct that cell's top edge, then
        // aim at its vertical center. Columns >= 5 have no slope offset.
        const int slope = col < 5 ? (4 - col) * 20 : 0;
        const int top_y = kLawnYMin + row * kRoofCellHeight + slope;
        out_y = top_y + kRoofCellHeight / 2;
    } else if (scene == 2 || scene == 3) { // pool/fog
        const int top_y = kLawnYMin + row * kPoolCellHeight;
        out_y = top_y + kPoolCellHeight / 2;
    } else { // day/night grass
        const int top_y = kLawnYMin + row * kGrassCellHeight;
        out_y = top_y + kGrassCellHeight / 2;
    }

    return out_x >= 0 && out_x < kBoardWidth && out_y >= 0 && out_y < kBoardHeight;
}

OriginalCobCallResult resolve_unlocked(PvzRuntime::VersionedPvZ& pvz,
                                       int source_row, int source_col,
                                       int target_row, int target_col) {
    if (!pvz.GameOn()) return result(ActionError::GameNotFound);
    if (pvz.GameUI() != 3) return result(ActionError::GameNotInBattleState);

    int target_x = -1;
    int target_y = -1;
    if (!grid_to_target_center(pvz, target_row, target_col, target_x, target_y)) {
        return result(ActionError::CobTargetOutOfArena);
    }

    int rows = pvz.GetRowCount();
    if (rows <= 0) rows = 5;
    if (source_row < 0 || source_row >= rows || source_col < 0 || source_col > 7) {
        return result(ActionError::CobCannonNotFound);
    }

    const uintptr_t board = pvz.ReadMemory<uintptr_t>({pvz.data().lawn, pvz.data().board});
    if (!board) return result(ActionError::GameNotInBattleState);
    const uintptr_t plants = pvz.ReadMemory<uintptr_t>({board + pvz.data().plant});
    int plant_max = pvz.ReadMemory<int>({board + pvz.data().plant_count_max});
    plant_max = std::max(0, std::min(1024, plant_max));

    for (int i = 0; plants && i < plant_max; ++i) {
        const uintptr_t p = plants + static_cast<uintptr_t>(i) * kPlantStructSize;
        if (pvz.ReadMemory<bool>({p + pvz.data().plant_dead})) continue;
        if (pvz.ReadMemory<int>({p + pvz.data().plant_type}) != 47) continue;
        if (pvz.ReadMemory<int>({p + pvz.data().plant_row}) != source_row ||
            pvz.ReadMemory<int>({p + pvz.data().plant_col}) != source_col) continue;
        if (pvz.ReadMemory<int>({p + PvzRuntime::kPlantStateOffset}) != PvzRuntime::kCobCannonReadyState) {
            return result(ActionError::CobCannonRecharging, i, target_x, target_y);
        }
        if (!PvzRuntime::cob_cannon_fire_address(pvz)) {
            return result(ActionError::UnsupportedGameVersion, i, target_x, target_y);
        }
        const auto identity = pvz.ReadMemory<std::uint32_t>({
            p + kPlantStructSize - sizeof(std::uint32_t)});
        return result(ActionError::None, i, target_x, target_y, identity);
    }
    return result(ActionError::CobCannonNotFound);
}
}  // namespace

OriginalCobCallResult CobCalls::resolve(int source_row, int source_col,
                                        int target_row, int target_col) {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return result(ActionError::GameNotFound);
    return resolve_unlocked(pvz, source_row, source_col, target_row, target_col);
}

OriginalCobCallResult CobCalls::resolve_next(int target_row, int target_col) {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return result(ActionError::GameNotFound);
    if (pvz.GameUI() != 3) return result(ActionError::GameNotInBattleState);
    int x = -1, y = -1;
    if (!grid_to_target_center(pvz, target_row, target_col, x, y))
        return result(ActionError::CobTargetOutOfArena);
    if (!PvzRuntime::cob_cannon_fire_address(pvz))
        return result(ActionError::UnsupportedGameVersion);
    const uintptr_t board = pvz.ReadMemory<uintptr_t>({pvz.data().lawn, pvz.data().board});
    if (!board) return result(ActionError::GameNotInBattleState);
    int rows = pvz.GetRowCount();
    if (rows <= 0) rows = 5;
    const auto ready = ready_cannons_unlocked(pvz, board, rows);
    if (ready.empty()) return result(ActionError::CobCannonRecharging);
    return result(ActionError::None, ready.front().id, x, y,
                  ready.front().identity);
}

bool CobCalls::build_flat_action_mask(std::vector<bool>& out_mask) {
    constexpr int kMaxRows = 6;
    constexpr int kTargetCols = 9;
    constexpr int kTargetCells = kMaxRows * kTargetCols;
    out_mask.assign(kTargetCells, false);

    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return false;
    if (pvz.GameUI() != 3) return true;
    if (!PvzRuntime::cob_cannon_fire_address(pvz)) return true;

    int rows = pvz.GetRowCount();
    if (rows <= 0) rows = 5;
    rows = std::min(rows, kMaxRows);

    const uintptr_t board = pvz.ReadMemory<uintptr_t>({pvz.data().lawn, pvz.data().board});
    if (!board) return false;
    if (ready_cannons_unlocked(pvz, board, rows).empty()) return true;

    bool valid_targets[kTargetCells] = {};
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < kTargetCols; ++col) {
            int x = -1;
            int y = -1;
            valid_targets[row * kTargetCols + col] =
                grid_to_target_center(pvz, row, col, x, y);
        }
    }

    for (int target = 0; target < kTargetCells; ++target) {
        out_mask[static_cast<std::size_t>(target)] = valid_targets[target];
    }
    return true;
}

OriginalCobCallResult CobCalls::fire(
    int cannon_id, std::uint32_t expected_identity, int target_x, int target_y) {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return result(ActionError::GameNotFound);
    if (pvz.GameUI() != 3) return result(ActionError::GameNotInBattleState);
    if (target_x < 0 || target_x >= kBoardWidth || target_y < 0 || target_y >= kBoardHeight) {
        return result(ActionError::CobTargetOutOfArena, cannon_id, target_x, target_y);
    }

    const uintptr_t board = pvz.ReadMemory<uintptr_t>({pvz.data().lawn, pvz.data().board});
    if (!board) return result(ActionError::GameNotInBattleState);
    const uintptr_t plants = pvz.ReadMemory<uintptr_t>({board + pvz.data().plant});
    int plant_max = pvz.ReadMemory<int>({board + pvz.data().plant_count_max});
    plant_max = std::max(0, std::min(1024, plant_max));
    if (!plants || cannon_id < 0 || cannon_id >= plant_max) {
        return result(ActionError::CobCannonNotFound);
    }

    const uintptr_t cannon = plants + static_cast<uintptr_t>(cannon_id) * kPlantStructSize;
    const std::uint32_t identity = pvz.ReadMemory<std::uint32_t>({
        cannon + kPlantStructSize - sizeof(std::uint32_t)});
    if (expected_identity == 0 || identity != expected_identity) {
        return result(ActionError::CobCannonNotFound);
    }
    if (pvz.ReadMemory<bool>({cannon + pvz.data().plant_dead}) ||
        pvz.ReadMemory<int>({cannon + pvz.data().plant_type}) != 47) {
        return result(ActionError::CobCannonNotFound);
    }
    if (pvz.ReadMemory<int>({cannon + PvzRuntime::kPlantStateOffset}) != PvzRuntime::kCobCannonReadyState) {
        return result(ActionError::CobCannonRecharging, cannon_id, target_x, target_y);
    }

    const uintptr_t fn = PvzRuntime::cob_cannon_fire_address(pvz);
    if (!fn) return result(ActionError::UnsupportedGameVersion, cannon_id, target_x, target_y);

    // Original Plant::CobCannonFire consumes board-local pixel coordinates.
    // They were derived from target_row/target_col above; the agent never sees them.
    pvz.asm_init();
    pvz.asm_push_dword(static_cast<unsigned int>(target_y));
    pvz.asm_push_dword(static_cast<unsigned int>(target_x));
    pvz.asm_mov_exx(Pt::Reg::EAX, static_cast<unsigned int>(cannon));
    pvz.asm_call(static_cast<unsigned int>(fn));
    pvz.asm_ret();
    pvz.asm_code_inject();

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    while (std::chrono::steady_clock::now() < deadline) {
        if (pvz.ReadMemory<int>({cannon + PvzRuntime::kPlantStateOffset}) != PvzRuntime::kCobCannonReadyState) {
            g_ready_since.erase(identity);
            return result(ActionError::None, cannon_id, target_x, target_y);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return result(ActionError::ActionTimeout, cannon_id, target_x, target_y);
}

}  // namespace pvz::game::original
