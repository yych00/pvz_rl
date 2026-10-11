#pragma once
// 待办：后续让感知层与 Gym 共用底层事实读取器。
#include "pvz/types.h"
#include "runtime/pvz_runtime_layout.h"

#include <cstdint>
#include <vector>

namespace pvz::game { struct RuntimeState; }

namespace pvz::observation {

// 内部无锁读取函数；调用者必须已持有 process_manager 互斥锁并提供有效的 pvz 与 board。
bool read_global_nolock(PvzRuntime::VersionedPvZ& pvz, uintptr_t board, GlobalState& out_global,
                        const game::RuntimeState* state = nullptr);
bool read_cards_nolock(PvzRuntime::VersionedPvZ& pvz, uintptr_t board, const GlobalState& global, std::vector<CardState>& out_cards);
bool read_plants_nolock(PvzRuntime::VersionedPvZ& pvz, uintptr_t board, int grid_rows, std::vector<PlantState>& out_plants);
bool read_zombies_nolock(PvzRuntime::VersionedPvZ& pvz, uintptr_t board, int grid_rows, std::vector<ZombieState>& out_zombies,
                         std::vector<ZombieState>* health = nullptr,
                         std::vector<BossProjectileState>* boss_projectiles = nullptr);
bool read_dropped_cards_nolock(PvzRuntime::VersionedPvZ& pvz, uintptr_t board, std::vector<DroppedCardState>& out_dropped_cards);
bool read_grid_items_nolock(PvzRuntime::VersionedPvZ& pvz, uintptr_t board, int grid_rows, std::vector<GridItemState>& out_grid_items);
bool read_ice_trails_nolock(PvzRuntime::VersionedPvZ& pvz, uintptr_t board, int grid_rows, std::vector<IceTrailState>& out_ice_trails);
bool read_mowers_nolock(PvzRuntime::VersionedPvZ& pvz, uintptr_t board, int grid_rows, std::vector<MowerState>& out_mowers);
bool read_grid_nolock(PvzRuntime::VersionedPvZ& pvz, uintptr_t board, int scene, int grid_rows,
                      const std::vector<PlantState>& plants, std::vector<GridSquareState>& out_grid);

}  // 命名空间 pvz::observation
