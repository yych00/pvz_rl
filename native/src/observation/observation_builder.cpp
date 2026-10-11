#include "observation_builder.h"
#include "reader_internal.h"
#include "../game/runtime_phase.h"
#include "../process/process_manager.h"
#include "../runtime/resident_executor.h"

#include <algorithm>
#include <chrono>
#include <thread>

namespace pvz::observation {

namespace {

inline int get_scene_rows(int scene) {
    return (scene == 2 || scene == 3) ? 6 : 5;
}

inline bool has_observable_board_scene(int ui) {
    // 僵尸胜利场景中权威 Board 仍存在，允许终局 Gym step 返回真实的最终观测。
    return ui == 3 || ui == 4 || ui == 5;
}

bool build_internal_nolock(PvzRuntime::VersionedPvZ& pvz, Observation& output) {
    pvz.ensure_remote_idle();
    Observation out_obs;
    const auto state = game::RuntimePhaseReader::read_nolock(pvz);
    if (!state.process_available || !has_observable_board_scene(state.raw_scene)) return false;
    const auto board = state.board_address;
    if (!board) return false;

    if (!read_global_nolock(pvz, board, out_obs.global, &state)) throw PvzRuntime::NativeOperationError(ActionError::Unsupported, "required observation component unavailable: global");
    int grid_rows = get_scene_rows(out_obs.global.scene);

    if (!read_cards_nolock(pvz, board, out_obs.global, out_obs.cards)) throw PvzRuntime::NativeOperationError(ActionError::Unsupported, "required observation component unavailable: cards");
    if (!read_plants_nolock(pvz, board, grid_rows, out_obs.plants)) throw PvzRuntime::NativeOperationError(ActionError::Unsupported, "required observation component unavailable: plants");
    if (!read_zombies_nolock(pvz, board, grid_rows, out_obs.zombies, &out_obs.zombie_health, &out_obs.boss_projectiles)) throw PvzRuntime::NativeOperationError(ActionError::Unsupported, "required observation component unavailable: zombies/boss_projectiles");
    if (!read_dropped_cards_nolock(pvz, board, out_obs.dropped_cards)) throw PvzRuntime::NativeOperationError(ActionError::Unsupported, "required observation component unavailable: dropped_cards");
    if (!read_grid_items_nolock(pvz, board, grid_rows, out_obs.grid_items)) throw PvzRuntime::NativeOperationError(ActionError::Unsupported, "required observation component unavailable: grid_items");
    if (!read_ice_trails_nolock(pvz, board, grid_rows, out_obs.ice_trails)) throw PvzRuntime::NativeOperationError(ActionError::Unsupported, "required observation component unavailable: ice_trails");
    if (!read_mowers_nolock(pvz, board, grid_rows, out_obs.mowers)) throw PvzRuntime::NativeOperationError(ActionError::Unsupported, "required observation component unavailable: mowers");
    if (!read_grid_nolock(pvz, board, out_obs.global.scene, grid_rows, out_obs.plants, out_obs.grid)) throw PvzRuntime::NativeOperationError(ActionError::Unsupported, "required observation component unavailable: grid");

    const auto final_state = game::RuntimePhaseReader::read_nolock(pvz);
    if (final_state.board_address != board || final_state.game_clock != state.game_clock ||
        final_state.raw_scene != state.raw_scene || final_state.phase != state.phase)
        throw PvzRuntime::NativeOperationError(ActionError::Unsupported,
                                              "Board, clock or runtime phase changed while reading observation");

    output = std::move(out_obs);
    return true;
}

class MainLoopSnapshotGuard {
public:
    explicit MainLoopSnapshotGuard(PvzRuntime::VersionedPvZ& pvz)
        : pvz_(pvz) {
        if (!pvz_.GameOn()) return;

        // block_main_loop 只作为远程读取期间的短同步屏障，完整观测读完后立即恢复主循环。
        pvz_.enable_hack(pvz_.data().block_main_loop, true);
        const int frame_ms = std::max(1, pvz_.GetFrameDuration());
        std::this_thread::sleep_for(std::chrono::milliseconds(frame_ms * 2));
        active_ = true;
    }

    ~MainLoopSnapshotGuard() {
        if (active_) {
            pvz_.enable_hack(pvz_.data().block_main_loop, false);
        }
    }

    bool active() const noexcept { return active_; }

private:
    PvzRuntime::VersionedPvZ& pvz_;
    bool active_ = false;
};

} // 匿名命名空间

bool ObservationBuilder::build(Observation& out_obs) {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();

    if (!pvz.GameOn() && !pvz.FindPvZ()) return false;
    pvz.ensure_remote_idle();
    if (!has_observable_board_scene(pvz.read_memory_checked<int>({pvz.data().lawn, pvz.data().game_ui}))) return false;

    uintptr_t board = pvz.read_memory_checked<uintptr_t>({pvz.data().lawn, pvz.data().board});
    if (!board) return false;

    const bool is_paused = pvz.read_memory_checked<bool>({board + pvz.data().game_paused});
    if (is_paused || runtime::ResidentExecutor::instance().owner()) {
        return build_internal_nolock(pvz, out_obs);
    }

    // 动画模式下，实体变多后一次完整观测读取可能跨过一个游戏刻，
    // 单纯重试直到前后时钟相等会越来越容易失败。这里短暂冻结原生主循环，
    // 一次性读取完整快照后立即恢复；API 调用之间不会保持冻结。
    MainLoopSnapshotGuard freeze(pvz);
    if (!freeze.active()) return false;
    if (!has_observable_board_scene(pvz.read_memory_checked<int>({pvz.data().lawn, pvz.data().game_ui}))) return false;
    if (pvz.read_memory_checked<uintptr_t>({pvz.data().lawn, pvz.data().board}) != board) return false;
    return build_internal_nolock(pvz, out_obs);
}

bool ObservationBuilder::read_global(GlobalState& out_global) {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return false;
    if (pvz.read_memory_checked<int>({pvz.data().lawn, pvz.data().game_ui}) != 3) return false;
    uintptr_t board = pvz.read_memory_checked<uintptr_t>({pvz.data().lawn, pvz.data().board});
    return read_global_nolock(pvz, board, out_global);
}

bool ObservationBuilder::read_cards(std::vector<CardState>& out_cards) {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return false;
    if (pvz.read_memory_checked<int>({pvz.data().lawn, pvz.data().game_ui}) != 3) return false;
    uintptr_t board = pvz.read_memory_checked<uintptr_t>({pvz.data().lawn, pvz.data().board});
    GlobalState global;
    if (!read_global_nolock(pvz, board, global)) return false;
    return read_cards_nolock(pvz, board, global, out_cards);
}

bool ObservationBuilder::read_plants(std::vector<PlantState>& out_plants) {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return false;
    if (pvz.read_memory_checked<int>({pvz.data().lawn, pvz.data().game_ui}) != 3) return false;
    uintptr_t board = pvz.read_memory_checked<uintptr_t>({pvz.data().lawn, pvz.data().board});
    int scene = pvz.read_memory_checked<int>({pvz.data().lawn, pvz.data().board, pvz.data().scene});
    return read_plants_nolock(pvz, board, get_scene_rows(scene), out_plants);
}

bool ObservationBuilder::read_zombies(std::vector<ZombieState>& out_zombies) {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return false;
    if (pvz.read_memory_checked<int>({pvz.data().lawn, pvz.data().game_ui}) != 3) return false;
    uintptr_t board = pvz.read_memory_checked<uintptr_t>({pvz.data().lawn, pvz.data().board});
    int scene = pvz.read_memory_checked<int>({pvz.data().lawn, pvz.data().board, pvz.data().scene});
    return read_zombies_nolock(pvz, board, get_scene_rows(scene), out_zombies);
}

bool ObservationBuilder::read_dropped_cards(std::vector<DroppedCardState>& out_dropped_cards) {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return false;
    if (pvz.read_memory_checked<int>({pvz.data().lawn, pvz.data().game_ui}) != 3) return false;
    uintptr_t board = pvz.read_memory_checked<uintptr_t>({pvz.data().lawn, pvz.data().board});
    return read_dropped_cards_nolock(pvz, board, out_dropped_cards);
}

bool ObservationBuilder::read_grid_items(std::vector<GridItemState>& out_grid_items) {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return false;
    if (pvz.read_memory_checked<int>({pvz.data().lawn, pvz.data().game_ui}) != 3) return false;
    uintptr_t board = pvz.read_memory_checked<uintptr_t>({pvz.data().lawn, pvz.data().board});
    int scene = pvz.read_memory_checked<int>({pvz.data().lawn, pvz.data().board, pvz.data().scene});
    return read_grid_items_nolock(pvz, board, get_scene_rows(scene), out_grid_items);
}

bool ObservationBuilder::read_ice_trails(std::vector<IceTrailState>& out_ice_trails) {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return false;
    if (pvz.read_memory_checked<int>({pvz.data().lawn, pvz.data().game_ui}) != 3) return false;
    uintptr_t board = pvz.read_memory_checked<uintptr_t>({pvz.data().lawn, pvz.data().board});
    int scene = pvz.read_memory_checked<int>({pvz.data().lawn, pvz.data().board, pvz.data().scene});
    return read_ice_trails_nolock(pvz, board, get_scene_rows(scene), out_ice_trails);
}

bool ObservationBuilder::read_mowers(std::vector<MowerState>& out_mowers) {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return false;
    if (pvz.read_memory_checked<int>({pvz.data().lawn, pvz.data().game_ui}) != 3) return false;
    uintptr_t board = pvz.read_memory_checked<uintptr_t>({pvz.data().lawn, pvz.data().board});
    int scene = pvz.read_memory_checked<int>({pvz.data().lawn, pvz.data().board, pvz.data().scene});
    return read_mowers_nolock(pvz, board, get_scene_rows(scene), out_mowers);
}

bool ObservationBuilder::read_grid(std::vector<GridSquareState>& out_grid) {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return false;
    if (pvz.read_memory_checked<int>({pvz.data().lawn, pvz.data().game_ui}) != 3) return false;
    uintptr_t board = pvz.read_memory_checked<uintptr_t>({pvz.data().lawn, pvz.data().board});
    int scene = pvz.read_memory_checked<int>({pvz.data().lawn, pvz.data().board, pvz.data().scene});
    int grid_rows = get_scene_rows(scene);
    std::vector<PlantState> plants;
    if (!read_plants_nolock(pvz, board, grid_rows, plants)) return false;
    return read_grid_nolock(pvz, board, scene, grid_rows, plants, out_grid);
}

}  // 命名空间 pvz::observation
