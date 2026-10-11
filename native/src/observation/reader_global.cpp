#include "reader_internal.h"
#include "../game/runtime_phase.h"
#include "runtime/conveyor_seed_bank.h"
#include <algorithm>
#include <cmath>

namespace pvz::observation {

bool read_global_nolock(PvzRuntime::VersionedPvZ& pvz, uintptr_t board, GlobalState& out_global,
                        const game::RuntimeState* supplied_state) {
    if (!board) return false;
    const auto state = supplied_state ? *supplied_state : game::RuntimePhaseReader::read_nolock(pvz);
    if (state.board_address != board || !state.game_clock_available) return false;
    out_global.board_id = static_cast<std::uint32_t>(board);
    out_global.raw_scene = state.raw_scene;
    out_global.runtime_phase = static_cast<std::int32_t>(state.phase);
    out_global.level_complete = state.level_complete;

    // 1. 读取逻辑时钟。
    out_global.game_clock = state.game_clock;

    // 2. 计算版本偏移差值，原版为 0，年度版/汉化年度版为 0x18。
    uintptr_t board_var_offset = pvz.data().sun - 0x5560;

    out_global.triggered_mowers = pvz.read_memory_checked<int>({board + 0x576C + board_var_offset});
    if (out_global.triggered_mowers < 0 || out_global.triggered_mowers > 10000)
        throw PvzRuntime::NativeOperationError(ActionError::Unsupported, "invalid triggered mower count");
    out_global.mower_trigger_waves.clear();
    for (uintptr_t row = 0; row < 6; ++row)
        out_global.mower_trigger_waves.push_back(pvz.read_memory_checked<int>(
            {board + 0x5F0 + board_var_offset + row * sizeof(std::int32_t)}));

    // 3. 阳光。
    out_global.sun = pvz.read_memory_checked<int>({board + pvz.data().sun});

    // 4. 场景与行数。
    out_global.scene = pvz.read_memory_checked<int>({board + pvz.data().scene});
    out_global.grid_rows = (out_global.scene == 2 || out_global.scene == 3) ? 6 : 5;
    out_global.fog_state_available = false;
    out_global.fog_offset = 0.0f;
    out_global.fog_alpha.clear();
    if (out_global.scene == 3) {
        // Retail Board: mGridCelFog=0x4C8, mFogOffset=0x5D0.
        // GOTY variants use the existing 0x18 Board delta, as for row_type.
        std::vector<std::int32_t> alpha(9 * 7);
        const float offset = pvz.read_memory_checked<float>({board + 0x5D0 + board_var_offset});
        if (!pvz.read_remote_bytes(board + 0x4C8 + board_var_offset,
                                   alpha.data(), alpha.size() * sizeof(std::int32_t)) ||
            !std::isfinite(offset) || offset < 0.0f || offset > 2000.0f ||
            std::any_of(alpha.begin(), alpha.end(), [](int a) { return a < 0 || a > 255; }))
            throw PvzRuntime::NativeOperationError(ActionError::Unsupported, "invalid fog snapshot");
        out_global.fog_alpha = std::move(alpha);
        out_global.fog_offset = offset;
        out_global.fog_state_available = true;
    }

    // 关卡编号只返回游戏内存中的真实值，不猜测或回退到 1-1。
    out_global.level_id = state.level_id;

    // 本关预定会出现的僵尸类型列表；空列表也是合法事实，不注入默认怪池。
    out_global.zombie_types_in_level.clear();
    for (int zid = 0; zid <= 32; ++zid) {
        if (pvz.read_memory_checked<bool>({board + 0x54D4 + board_var_offset + (uintptr_t)zid})) {
            out_global.zombie_types_in_level.push_back(zid);
        }
    }

    // 5. 波次信息保留原始事实，不用常量 20 修补异常读取。
    out_global.total_waves = pvz.read_memory_checked<int>({board + 0x5564 + board_var_offset});
    if (out_global.total_waves < 0 || out_global.total_waves > 100)
        throw PvzRuntime::NativeOperationError(ActionError::Unsupported, "invalid total wave count");
    out_global.current_wave = pvz.read_memory_checked<int>({board + 0x557C + board_var_offset});
    out_global.wave_countdown = pvz.read_memory_checked<int>({board + 0x559C + board_var_offset});

    // Board.h: mZombiesInWave[100][50] at 0x6B4 (GOTY uses versioned offset).
    // A checked block read avoids interpreting a failed read as normal zombies.
    out_global.zombie_spawn_plan.clear();
    out_global.spawn_plan_available = false;
    if (out_global.total_waves > 0 && out_global.total_waves <= 100) {
        std::vector<std::int32_t> plan(static_cast<size_t>(out_global.total_waves) * 50);
        if (pvz.read_remote_bytes(board + pvz.data().spawn_list,
                                  plan.data(), plan.size() * sizeof(std::int32_t))) {
            bool valid = true;
            for (size_t start = 0; start < plan.size() && valid; start += 50) {
                bool ended = false;
                for (size_t i = start; i < start + 50; ++i) {
                    // Only the first -1 is initialized as the row terminator.
                    // Do not interpret stale memory beyond that sentinel.
                    if (ended) { plan[i] = -1; continue; }
                    if (plan[i] == -1) { ended = true; continue; }
                    if (plan[i] < 0 || plan[i] > 32) { valid = false; break; }
                }
            }
            if (!valid) throw PvzRuntime::NativeOperationError(ActionError::Unsupported, "invalid zombie spawn plan");
            if (valid) {
                out_global.zombie_spawn_plan = std::move(plan);
                out_global.spawn_plan_available = true;
            }
        } else {
            throw PvzRuntime::NativeOperationError(ActionError::Unsupported, "zombie spawn plan read failed");
        }
    }

    // 6. 根据真实波次事实判断旗帜大波。
    const int current_wave = out_global.current_wave;
    const int total_waves = out_global.total_waves;
    out_global.is_huge_wave = (current_wave > 0) &&
        (current_wave == total_waves ||
         (total_waves >= 20 && current_wave == 10) ||
         (total_waves >= 30 && (current_wave == 10 || current_wave == 20)));

    // 7. 暂停状态。
    out_global.is_paused = pvz.read_memory_checked<bool>({board + pvz.data().game_paused});

    // ObservationBuilder 只会在可观测棋盘场景进入这里，但状态仍统一交给运行阶段分类器，
    // 避免把原始场景值 2 错误地直接等同于选卡阶段。
    out_global.status = state.status;

    // 8. 游戏模式与卡牌供给系统。
    out_global.game_mode = state.game_mode;
    out_global.game_mode_name = runtime::game_mode_name_nolock(out_global.game_mode, out_global.level_id);
    out_global.is_conveyor = runtime::has_conveyor_seed_bank_nolock(pvz, board);
    out_global.card_system = runtime::card_system_type_nolock(pvz, board);

    return true;
}

}  // 命名空间 pvz::observation
