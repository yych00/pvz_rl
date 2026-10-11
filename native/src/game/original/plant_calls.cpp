#include "runtime/remote_allocation.h"
#include "plant_calls.h"
#include "../../runtime/resident_executor.h"
#include "../simulation_clock.h"

#include "../../process/process_manager.h"
#include "../card_sources.h"
#include "runtime/conveyor_seed_bank.h"
#include "runtime/pvz_runtime_layout.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <vector>
#include <Windows.h>

namespace pvz::game::original {
namespace {

constexpr uintptr_t kPlantStructSize = 0x14C;

constexpr uintptr_t kPlantHealthOffset = 0x40;
constexpr uintptr_t kPlantMaxHealthOffset = 0x44;
constexpr uintptr_t kWakeUpCounterOffset = 0x130;
constexpr uintptr_t kOnBungeeStateOffset = 0x134;
constexpr uintptr_t kSleepingOffset = 0x143;

constexpr uintptr_t kClassicPlayerPurchasesOffset = 0x1C0;
constexpr int kStoreItemFirstAid = 29;

constexpr int kBungeeGettingGrabbed = 1;
constexpr int kBungeeRising = 2;

constexpr int kMaskSlots = 10;
constexpr int kMaskRows = 6;
constexpr int kMaskCols = 9;
constexpr int kPlantMaskSize = kMaskSlots * kMaskRows * kMaskCols;
constexpr std::uint32_t kNativeReasonSentinel = 0xFFFFFFFFu;
constexpr std::uint32_t kMaxPlantingReason = 13u;

struct PlantContext {
    uintptr_t board = 0;
    uintptr_t seed_bank = 0;
    game::CardSourceKind source = game::CardSourceKind::None;
    uintptr_t coin = 0;
    int plant_type = -1;
    int actual_plant_type = -1;
    bool imitater = false;
    int sun = 0;
    int cost = 0;
    int cooldown = 0;

    uintptr_t carrier = 0;
    uintptr_t main_plant = 0;
    uintptr_t pumpkin = 0;
    uintptr_t kernel_left = 0;
    uintptr_t kernel_right = 0;
    uintptr_t first_aid_replaced_plant = 0;
};

struct CellPlants {
    uintptr_t under = 0;
    int under_raw_type = -1;
    int under_type = -1;
    int under_bungee = 0;

    uintptr_t normal = 0;
    int normal_raw_type = -1;
    int normal_type = -1;
    int normal_bungee = 0;
    int normal_health = 0;
    int normal_max_health = 0;
    bool normal_sleeping = false;
    int normal_wake_up = 0;

    uintptr_t pumpkin = 0;
    int pumpkin_bungee = 0;
    int pumpkin_health = 0;
    int pumpkin_max_health = 0;

    uintptr_t flying = 0;
};

struct NativeSlotGate {
    int plant_type = -1;
    int actual_plant_type = -1;
    ActionError error = ActionError::Unsupported;
};

struct NativeGlobalContext {
    uintptr_t board = 0;
    int rows = 5;
    int slot_count = 0;
    int sun = 0;
    ActionError error = ActionError::Unsupported;
    std::array<NativeSlotGate, kMaskSlots> slots{};
};

struct NativePlantQuery {
    int offset = -1;
    int slot = -1;
    int row = -1;
    int col = -1;
    int plant_type = -1;
    uintptr_t board = 0;
};

struct NativeLegalityResult {
    bool available = false;
    bool accepted = false;
    bool original_called = false;
    int plant_type = -1;
    int planting_reason = -1;
    ActionError error = ActionError::Unsupported;
};

struct CompareStats {
    std::uint64_t actions_seen = 0;
    std::uint64_t compared = 0;
    std::uint64_t matched = 0;
    std::uint64_t mismatched = 0;
    std::uint64_t native_unavailable = 0;
    std::uint64_t original_calls = 0;
    std::uint64_t mask_batches = 0;
    std::uint64_t single_checks = 0;
    std::uint64_t consecutive_matches = 0;
    uintptr_t last_native_address = 0;

    bool has_last_mismatch = false;
    int last_slot = -1;
    int last_row = -1;
    int last_col = -1;
    int last_plant_type = -1;
    int last_legacy_error = static_cast<int>(ActionError::Unsupported);
    int last_native_error = static_cast<int>(ActionError::Unsupported);
    int last_planting_reason = -1;
    bool last_legacy_accepted = false;
    bool last_native_accepted = false;
};

std::mutex g_compare_mutex;
CompareStats g_compare_stats;

bool is_upgrade(int type) {
    return type >= 40 && type <= 47;
}

int upgrade_base(int type) {
    switch (type) {
    case 40: return 7;
    case 41: return 1;
    case 42: return 10;
    case 43: return 16;
    case 44: return 39;
    case 45: return 31;
    case 46: return 21;
    case 47: return 34;
    default: return -1;
    }
}

int actual_existing_type(PvzRuntime::VersionedPvZ& pvz, uintptr_t plant, int raw_type) {
    if (raw_type != 48) return raw_type;
    const int imitater = pvz.ReadMemory<int>({plant + pvz.data().plant_imitater});
    return imitater >= 0 ? imitater : raw_type;
}

bool is_not_on_ground(PvzRuntime::VersionedPvZ& pvz, uintptr_t plant) {
    if (!plant) return true;
    if (pvz.ReadMemory<bool>({plant + pvz.data().plant_dead})) return true;
    if (pvz.ReadMemory<bool>({plant + pvz.data().plant_squished})) return true;
    return pvz.ReadMemory<int>({plant + kOnBungeeStateOffset}) == kBungeeRising;
}

bool has_first_aid(PvzRuntime::VersionedPvZ& pvz) {
    if (pvz.version_id() != PVZ_1_0_0_1051_EN &&
        pvz.version_id() != PVZ_1_2_0_1065_EN) {
        return false;
    }
    const uintptr_t user =
        pvz.ReadMemory<uintptr_t>({pvz.data().lawn, pvz.data().user_data});
    if (!user) return false;
    return pvz.ReadMemory<int>({
        user + kClassicPlayerPurchasesOffset +
        static_cast<uintptr_t>(kStoreItemFirstAid) * sizeof(int)}) > 0;
}

bool easy_planting(PvzRuntime::VersionedPvZ& pvz) {
    return pvz.ReadMemory<int>({pvz.data().lawn, pvz.data().free_planting}) != 0;
}

int grid_square_type(PvzRuntime::VersionedPvZ& pvz, uintptr_t board, int row, int col) {
    return pvz.ReadMemory<int>({
        board + pvz.data().block_type +
        static_cast<uintptr_t>(col) * 0x18 +
        static_cast<uintptr_t>(row) * sizeof(int)});
}

bool is_ice_at(PvzRuntime::VersionedPvZ& pvz, uintptr_t board, int row, int col) {
    const uintptr_t ice_timer = board + pvz.data().ice_trail_cd;
    const int timer = pvz.ReadMemory<int>({ice_timer + static_cast<uintptr_t>(row) * 4});
    if (timer == 0) return false;

    const int min_x = pvz.ReadMemory<int>({
        ice_timer + static_cast<uintptr_t>(row) * 4 - 24});
    if (min_x > 750) return false;

    int first_ice_col = (min_x + 12 - 40) / 80;
    first_ice_col = std::max(0, std::min(8, first_ice_col));
    return col >= first_ice_col;
}

void scan_cell(
    PvzRuntime::VersionedPvZ& pvz, uintptr_t plants, int plant_max,
    int row, int col, CellPlants& out) {
    for (int i = 0; i < plant_max; ++i) {
        const uintptr_t p = plants + static_cast<uintptr_t>(i) * kPlantStructSize;
        if (is_not_on_ground(pvz, p)) continue;
        if (pvz.ReadMemory<int>({p + pvz.data().plant_row}) != row) continue;

        const int p_col = pvz.ReadMemory<int>({p + pvz.data().plant_col});
        const int raw = pvz.ReadMemory<int>({p + pvz.data().plant_type});
        const int type = actual_existing_type(pvz, p, raw);

        const bool occupies_target =
            p_col == col || (type == 47 && p_col + 1 == col);
        if (!occupies_target) continue;

        if (type == 35) {
            out.flying = p;
        } else if (type == 16 || type == 33) {
            out.under = p;
            out.under_raw_type = raw;
            out.under_type = type;
            out.under_bungee = pvz.ReadMemory<int>({p + kOnBungeeStateOffset});
        } else if (type == 30) {
            out.pumpkin = p;
            out.pumpkin_bungee = pvz.ReadMemory<int>({p + kOnBungeeStateOffset});
            out.pumpkin_health = pvz.ReadMemory<int>({p + kPlantHealthOffset});
            out.pumpkin_max_health = pvz.ReadMemory<int>({p + kPlantMaxHealthOffset});
        } else {
            out.normal = p;
            out.normal_raw_type = raw;
            out.normal_type = type;
            out.normal_bungee = pvz.ReadMemory<int>({p + kOnBungeeStateOffset});
            out.normal_health = pvz.ReadMemory<int>({p + kPlantHealthOffset});
            out.normal_max_health = pvz.ReadMemory<int>({p + kPlantMaxHealthOffset});
            out.normal_sleeping = pvz.ReadMemory<bool>({p + kSleepingOffset});
            out.normal_wake_up = pvz.ReadMemory<int>({p + kWakeUpCounterOffset});
        }
    }
}

bool valid_cob_spot(
    PvzRuntime::VersionedPvZ& pvz, uintptr_t plants, int plant_max,
    int row, int left_col, uintptr_t& out_left, uintptr_t& out_right) {
    if (left_col < 0 || left_col >= 8) return false;

    CellPlants left;
    CellPlants right;
    scan_cell(pvz, plants, plant_max, row, left_col, left);
    scan_cell(pvz, plants, plant_max, row, left_col + 1, right);

    if (left.pumpkin || right.pumpkin) return false;
    if (!left.normal || !right.normal ||
        left.normal_type != 34 || right.normal_type != 34) {
        return false;
    }
    if (left.normal_bungee == kBungeeGettingGrabbed ||
        right.normal_bungee == kBungeeGettingGrabbed) {
        return false;
    }

    const bool left_pot = left.under && left.under_type == 33 &&
                          left.under_bungee != kBungeeGettingGrabbed;
    const bool right_pot = right.under && right.under_type == 33 &&
                           right.under_bungee != kBungeeGettingGrabbed;
    if (left_pot != right_pot) return false;

    out_left = left.normal;
    out_right = right.normal;
    return true;
}

// Mirrors SeedBank::RemoveSeed: conveyor packets move left after use, while
// the game continues to add new packets at the first empty slot.
void consume_conveyor_seed_unlocked(
    PvzRuntime::VersionedPvZ& pvz, uintptr_t seed_bank, int slot) {
    const int count = std::max(0, std::min(10,
        pvz.ReadMemory<int>({seed_bank + pvz.data().slot_count})));
    for (int i = slot; i < count; ++i) {
        const uintptr_t packet = seed_bank + static_cast<uintptr_t>(i) * 0x50;
        if (pvz.ReadMemory<int>({packet + pvz.data().slot_seed_type}) == -1) {
            break;
        }

        if (i == count - 1) {
            pvz.WriteMemory<int>({-1}, {packet + pvz.data().slot_seed_type});
            pvz.WriteMemory<int>({-1}, {packet + pvz.data().slot_seed_type_im});
            pvz.WriteMemory<int>({0}, {packet + 0x58});
            pvz.WriteMemory<int>({0}, {packet + pvz.data().slot_seed_cd_past});
            pvz.WriteMemory<int>({0}, {packet + pvz.data().slot_seed_cd_total});
            pvz.WriteMemory<bool>({false}, {packet + 0x71});
            pvz.WriteMemory<bool>({true}, {packet + 0x70});
        } else {
            const uintptr_t next = packet + 0x50;
            const int next_type = pvz.ReadMemory<int>({
                next + pvz.data().slot_seed_type});
            const int next_imitater_type = next_type == -1 ? -1 :
                pvz.ReadMemory<int>({next + pvz.data().slot_seed_type_im});
            const int next_offset = pvz.ReadMemory<int>({next + 0x58});
            const int next_cd_past = pvz.ReadMemory<int>({
                next + pvz.data().slot_seed_cd_past});
            const int next_cd_total = pvz.ReadMemory<int>({
                next + pvz.data().slot_seed_cd_total});
            const bool next_refreshing = pvz.ReadMemory<bool>({next + 0x71});
            const bool next_active = pvz.ReadMemory<bool>({next + 0x70});
            pvz.WriteMemory<int>({next_type}, {packet + pvz.data().slot_seed_type});
            pvz.WriteMemory<int>({next_imitater_type},
                {packet + pvz.data().slot_seed_type_im});
            pvz.WriteMemory<int>({next_offset + 51}, {packet + 0x58});
            pvz.WriteMemory<int>({next_cd_past},
                {packet + pvz.data().slot_seed_cd_past});
            pvz.WriteMemory<int>({next_cd_total},
                {packet + pvz.data().slot_seed_cd_total});
            pvz.WriteMemory<bool>({next_refreshing}, {packet + 0x71});
            pvz.WriteMemory<bool>({next_active}, {packet + 0x70});
        }
    }
}

ActionError build_card_context_unlocked(
    PvzRuntime::VersionedPvZ& pvz, int slot, int row, int col,
    PlantContext& ctx) {
    if (!pvz.GameOn()) return ActionError::GameNotFound;
    if (pvz.GameUI() != 3) return ActionError::GameNotInBattleState;

    int rows = pvz.GetRowCount();
    if (rows <= 0) rows = 5;
    if (row < 0 || row >= rows || col < 0 || col >= 9) {
        return ActionError::InvalidGridCoordinates;
    }

    ctx.board = pvz.ReadMemory<uintptr_t>({pvz.data().lawn, pvz.data().board});
    if (!ctx.board) return ActionError::GameNotInBattleState;

    const int sun = pvz.ReadMemory<int>({ctx.board + pvz.data().sun});
    game::CardSlotFact card;
    if (!game::read_card_source_nolock(pvz, ctx.board, sun, slot, card)) {
        return ActionError::SlotIndexOutOfRange;
    }
    ctx.source = card.source;
    ctx.seed_bank = card.seed_bank;
    ctx.coin = card.coin;
    ctx.plant_type = card.raw_type;
    ctx.actual_plant_type = card.plant_type;
    ctx.imitater = card.imitater;
    ctx.sun = card.sun;
    ctx.cost = card.cost;
    ctx.cooldown = card.cooldown_on_use;
    return card.gate;
}

ActionError validate_plant_context_unlocked(
    PvzRuntime::VersionedPvZ& pvz, int row, int col,
    PlantContext& ctx, PlantContext* out_ctx) {
    const uintptr_t plants =
        pvz.ReadMemory<uintptr_t>({ctx.board + pvz.data().plant});
    int plant_max =
        pvz.ReadMemory<int>({ctx.board + pvz.data().plant_count_max});
    plant_max = std::max(0, std::min(1024, plant_max));

    CellPlants cell;
    if (plants) scan_cell(pvz, plants, plant_max, row, col, cell);
    ctx.carrier = cell.under;
    ctx.main_plant = cell.normal;
    ctx.pumpkin = cell.pumpkin;

    bool has_grave = false;
    bool has_crater = false;
    bool has_scary_pot = false;
    const uintptr_t items =
        pvz.ReadMemory<uintptr_t>({ctx.board + pvz.data().grid_item});
    int item_max =
        pvz.ReadMemory<int>({ctx.board + pvz.data().grid_item_count_max});
    item_max = std::max(0, std::min(128, item_max));
    if (items) {
        for (int i = 0; i < item_max; ++i) {
            const uintptr_t item =
                items + static_cast<uintptr_t>(i) * PvzRuntime::grid_item_struct_size(pvz);
            if (pvz.ReadMemory<bool>({item + pvz.data().grid_item_dead})) continue;
            if (pvz.ReadMemory<int>({item + pvz.data().grid_item_row}) != row ||
                pvz.ReadMemory<int>({item + pvz.data().grid_item_col}) != col) {
                continue;
            }
            const int type = pvz.ReadMemory<int>({item + pvz.data().grid_item_type});
            has_grave |= type == 1;
            has_crater |= type == 2;
            has_scary_pot |= type == 7;
        }
    }

    if (ctx.actual_plant_type == 11) {
        if (cell.normal) return ActionError::MainPlantAlreadyExists;
        if (!has_grave) return ActionError::InvalidGridCoordinates;
        if (out_ctx) *out_ctx = ctx;
        return ActionError::None;
    }

    if (ctx.actual_plant_type == 35) {
        if (cell.flying) return ActionError::MainPlantAlreadyExists;
        // Original Board::CanPlantAt does not whitelist seed types here. It
        // only requires a normal plant that is currently asleep, has not
        // started waking, and is not being grabbed by a Bungee. In normal
        // play that naturally selects mushrooms, including Gloom-shroom.
        if (!cell.normal || !cell.normal_sleeping || cell.normal_wake_up > 0 ||
            cell.normal_bungee == kBungeeGettingGrabbed) {
            return ActionError::InvalidGridCoordinates;
        }
        if (out_ctx) *out_ctx = ctx;
        return ActionError::None;
    }

    if (has_grave) return ActionError::GridBlockedByHazard;
    if (has_crater || has_scary_pot || is_ice_at(pvz, ctx.board, row, col)) {
        return ActionError::GridBlockedByHazard;
    }

    const int square = grid_square_type(pvz, ctx.board, row, col);
    if (square == 0 || square == 2) return ActionError::InvalidGridCoordinates;
    const bool pool = square == 3;
    const int scene = pvz.GetScene();
    const bool roof = scene == 4 || scene == 5;

    const bool has_lily =
        cell.under && cell.under_type == 16 &&
        cell.under_bungee != kBungeeGettingGrabbed;
    const bool has_pot =
        cell.under && cell.under_type == 33 &&
        cell.under_bungee != kBungeeGettingGrabbed;

    if (ctx.actual_plant_type == 16 ||
        ctx.actual_plant_type == 19 ||
        ctx.actual_plant_type == 24) {
        if (!pool) return ActionError::InvalidGridCoordinates;
        if (cell.normal || cell.under) return ActionError::MainPlantAlreadyExists;
        if (out_ctx) *out_ctx = ctx;
        return ActionError::None;
    }

    if (ctx.actual_plant_type == 21 || ctx.actual_plant_type == 46) {
        if (pool || roof || cell.under) return ActionError::InvalidGridCoordinates;
    }

    if (pool && !has_lily && ctx.actual_plant_type != 43) {
        const bool pumpkin_on_cattail =
            ctx.actual_plant_type == 30 &&
            cell.normal && cell.normal_type == 43;
        if (!pumpkin_on_cattail) return ActionError::TerrainRequiresLilyPad;
    }

    if (ctx.actual_plant_type == 33) {
        if (cell.normal || cell.under || cell.pumpkin) {
            return ActionError::MainPlantAlreadyExists;
        }
        if (out_ctx) *out_ctx = ctx;
        return ActionError::None;
    }

    if (roof && !has_pot) return ActionError::TerrainRequiresFlowerPot;

    if (ctx.actual_plant_type == 30) {
        if (cell.normal && cell.normal_type == 47) {
            return ActionError::InvalidGridCoordinates;
        }
        if (!cell.pumpkin) {
            if (out_ctx) *out_ctx = ctx;
            return ActionError::None;
        }
        if (has_first_aid(pvz) &&
            cell.pumpkin_max_health > 0 &&
            cell.pumpkin_health < cell.pumpkin_max_health * 2 / 3 &&
            cell.pumpkin_bungee != kBungeeGettingGrabbed) {
            ctx.first_aid_replaced_plant = cell.pumpkin;
            if (out_ctx) *out_ctx = ctx;
            return ActionError::None;
        }
        return ActionError::PumpkinAlreadyEquipped;
    }

    if (ctx.actual_plant_type == 4 && has_lily) {
        return ActionError::InvalidGridCoordinates;
    }

    if (cell.under) {
        if (ctx.actual_plant_type == 43) {
            if (cell.normal) return ActionError::MainPlantAlreadyExists;
            if (cell.under_raw_type != 48 &&
                cell.under_type == 16 &&
                cell.under_bungee != kBungeeGettingGrabbed) {
                if (out_ctx) *out_ctx = ctx;
                return ActionError::None;
            }
            return ActionError::UpgradeBaseMismatch;
        }
        if (cell.under_raw_type == 48) {
            return ActionError::InvalidGridCoordinates;
        }
    }

    if (cell.normal) {
        if (is_upgrade(ctx.actual_plant_type)) {
            if (ctx.actual_plant_type == 47) {
                uintptr_t left = 0;
                uintptr_t right = 0;
                if (cell.normal_type != 34 ||
                    !valid_cob_spot(pvz, plants, plant_max, row, col, left, right)) {
                    return ActionError::CobCannonDoubleBaseInvalid;
                }
                ctx.kernel_left = left;
                ctx.kernel_right = right;
                if (out_ctx) *out_ctx = ctx;
                return ActionError::None;
            }

            if (cell.normal_type == upgrade_base(ctx.actual_plant_type) &&
                cell.normal_bungee != kBungeeGettingGrabbed) {
                if (out_ctx) *out_ctx = ctx;
                return ActionError::None;
            }
            return ActionError::UpgradeBaseMismatch;
        }

        if ((ctx.actual_plant_type == 3 || ctx.actual_plant_type == 23) &&
            has_first_aid(pvz) &&
            cell.normal_type == ctx.actual_plant_type &&
            cell.normal_max_health > 0 &&
            cell.normal_health < cell.normal_max_health * 2 / 3 &&
            cell.normal_bungee != kBungeeGettingGrabbed) {
            ctx.first_aid_replaced_plant = cell.normal;
            if (out_ctx) *out_ctx = ctx;
            return ActionError::None;
        }
        return ActionError::MainPlantAlreadyExists;
    }

    if (!easy_planting(pvz) && is_upgrade(ctx.actual_plant_type)) {
        return ActionError::UpgradeBaseMismatch;
    }

    if (ctx.actual_plant_type == 43 && !pool) {
        return ActionError::InvalidGridCoordinates;
    }

    if (out_ctx) *out_ctx = ctx;
    return ActionError::None;
}

ActionError validate_unlocked(
    PvzRuntime::VersionedPvZ& pvz, int slot, int row, int col,
    PlantContext* out_ctx) {
    PlantContext ctx;
    const ActionError gate =
        build_card_context_unlocked(pvz, slot, row, col, ctx);
    if (gate != ActionError::None) return gate;
    return validate_plant_context_unlocked(pvz, row, col, ctx, out_ctx);
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

void put_imitater(
    PvzRuntime::VersionedPvZ& pvz, int row, int col, int target_type) {
    pvz.asm_init();
    pvz.asm_push_dword(target_type);
    pvz.asm_push_dword(48);
#ifdef _PVZ_BETA_LEAK_SUPPORT
    if (pvz.isBETA()) {
        pvz.asm_push_dword(row);
        pvz.asm_push_dword(col);
        pvz.asm_mov_exx_dword_ptr(Pt::Reg::ECX, pvz.data().lawn);
        pvz.asm_mov_exx_dword_ptr_exx_add(Pt::Reg::ECX, pvz.data().board);
    } else
#endif
    {
        pvz.asm_mov_exx(Pt::Reg::EAX, row);
        pvz.asm_push_dword(col);
        pvz.asm_mov_exx_dword_ptr(Pt::Reg::EBP, pvz.data().lawn);
        pvz.asm_mov_exx_dword_ptr_exx_add(Pt::Reg::EBP, pvz.data().board);
        pvz.asm_push_exx(Pt::Reg::EBP);
    }
    pvz.asm_call(pvz.data().call_put_plant);
    pvz.asm_ret();
    pvz.asm_code_inject();
}

bool is_executable_remote_address(
    PvzRuntime::VersionedPvZ& pvz, uintptr_t address) {
    if (!address || !pvz.process_handle()) return false;
    MEMORY_BASIC_INFORMATION mbi{};
    if (::VirtualQueryEx(
            pvz.process_handle(),
            reinterpret_cast<LPCVOID>(address),
            &mbi,
            sizeof(mbi)) == 0) {
        return false;
    }
    if (mbi.State != MEM_COMMIT || (mbi.Protect & PAGE_GUARD) != 0 ||
        (mbi.Protect & PAGE_NOACCESS) != 0) {
        return false;
    }
    const DWORD executable =
        PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
        PAGE_EXECUTE_WRITECOPY;
    return (mbi.Protect & executable) != 0;
}

ActionError build_native_global_context_unlocked(
    PvzRuntime::VersionedPvZ& pvz, NativeGlobalContext& ctx) {
    if (!pvz.GameOn()) return ActionError::GameNotFound;
    if (pvz.GameUI() != 3) return ActionError::GameNotInBattleState;

    ctx.rows = pvz.GetRowCount();
    if (ctx.rows <= 0) ctx.rows = 5;

    ctx.board = pvz.ReadMemory<uintptr_t>({pvz.data().lawn, pvz.data().board});
    if (!ctx.board) return ActionError::GameNotInBattleState;

    ctx.slot_count = runtime::seed_bank_slot_count_nolock(pvz, ctx.board);
    if (ctx.slot_count == 0) return ActionError::SlotIndexOutOfRange;
    ctx.sun = pvz.ReadMemory<int>({ctx.board + pvz.data().sun});
    return ActionError::None;
}

NativeSlotGate read_native_slot_gate_unlocked(
    PvzRuntime::VersionedPvZ& pvz,
    const NativeGlobalContext& ctx,
    int slot) {
    NativeSlotGate gate;
    if (slot < 0 || slot >= ctx.slot_count) {
        gate.error = ActionError::SlotIndexOutOfRange;
        return gate;
    }

    game::CardSlotFact card;
    if (!game::read_card_source_nolock(
            pvz, ctx.board, ctx.sun, slot, card)) {
        gate.error = ActionError::SlotIndexOutOfRange;
        return gate;
    }
    gate.plant_type = card.raw_type;
    gate.actual_plant_type = card.plant_type;
    gate.error = card.gate;
    return gate;
}

bool execute_native_can_plant_batch_unlocked(
    PvzRuntime::VersionedPvZ& pvz,
    const std::vector<NativePlantQuery>& queries,
    std::vector<int>& out_reasons,
    bool main_loop_already_frozen) {
    out_reasons.assign(queries.size(), -1);
    if (queries.empty()) return true;

    const uintptr_t call_addr = PvzRuntime::can_plant_at_address(pvz);
    if (!is_executable_remote_address(pvz, call_addr)) return false;

    const SIZE_T result_bytes = queries.size() * sizeof(std::uint32_t);
    LPVOID remote_results = ::VirtualAllocEx(
        pvz.process_handle(), nullptr, result_bytes,
        MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!remote_results) return false;
    PvzRuntime::RemoteAllocation allocation(pvz.process_handle(), remote_results);

    std::vector<std::uint32_t> remote_values(
        queries.size(), kNativeReasonSentinel);
    SIZE_T written = 0;
    const BOOL wrote = ::WriteProcessMemory(
        pvz.process_handle(), remote_results,
        remote_values.data(), result_bytes, &written);
    if (!wrote || written != result_bytes) {
        return false;
    }

    pvz.asm_init();
    const uintptr_t result_base =
        reinterpret_cast<uintptr_t>(remote_results);
    for (std::size_t i = 0; i < queries.size(); ++i) {
        const auto& q = queries[i];
        pvz.asm_push_dword(static_cast<unsigned int>(q.plant_type));
        pvz.asm_push_dword(static_cast<unsigned int>(q.col));
        pvz.asm_push_dword(static_cast<unsigned int>(q.board));
        pvz.asm_mov_exx(
            Pt::Reg::EAX, static_cast<unsigned int>(q.row));
        pvz.asm_call(static_cast<unsigned int>(call_addr));

        pvz.asm_add_byte(0xA3);
        pvz.asm_add_dword(static_cast<unsigned int>(
            result_base + i * sizeof(std::uint32_t)));
    }
    pvz.asm_add_list(0x31, 0xC0);
    pvz.asm_ret();

    pvz.asm_code_inject(main_loop_already_frozen);

    SIZE_T read = 0;
    const BOOL read_ok = ::ReadProcessMemory(
        pvz.process_handle(), remote_results,
        remote_values.data(), result_bytes, &read);
    if (!read_ok || read != result_bytes) return false;

    for (std::size_t i = 0; i < remote_values.size(); ++i) {
        const std::uint32_t reason = remote_values[i];
        out_reasons[i] =
            (reason == kNativeReasonSentinel ||
             reason > kMaxPlantingReason)
                ? -1
                : static_cast<int>(reason);
    }
    return true;
}

NativeLegalityResult query_native_legality_unlocked(
    PvzRuntime::VersionedPvZ& pvz, int slot, int row, int col,
    bool main_loop_already_frozen) {
    NativeLegalityResult out;

    NativeGlobalContext global;
    global.error = build_native_global_context_unlocked(pvz, global);
    if (global.error != ActionError::None) {
        out.available = true;
        out.error = global.error;
        return out;
    }
    if (row < 0 || row >= global.rows || col < 0 || col >= kMaskCols) {
        out.available = true;
        out.error = ActionError::InvalidGridCoordinates;
        return out;
    }

    const NativeSlotGate gate =
        read_native_slot_gate_unlocked(pvz, global, slot);
    out.plant_type = gate.actual_plant_type;
    if (gate.error != ActionError::None) {
        out.available = true;
        out.error = gate.error;
        return out;
    }

    NativePlantQuery query;
    query.offset = 0;
    query.slot = slot;
    query.row = row;
    query.col = col;
    query.plant_type = gate.actual_plant_type;
    query.board = global.board;

    std::vector<int> reasons;
    if (!execute_native_can_plant_batch_unlocked(
            pvz, std::vector<NativePlantQuery>{query},
            reasons, main_loop_already_frozen) ||
        reasons.size() != 1 || reasons[0] < 0) {
        out.available = false;
        out.original_called = true;
        out.error = ActionError::UnsupportedGameVersion;
        return out;
    }

    out.available = true;
    out.original_called = true;
    out.planting_reason = reasons[0];
    out.accepted = reasons[0] == 0;
    out.error = out.accepted
        ? ActionError::None
        : ActionError::InvalidGridCoordinates;
    return out;
}

void record_comparison(
    int slot, int row, int col, int plant_type,
    ActionError legacy_error, const NativeLegalityResult& native,
    uintptr_t native_address, bool from_mask) {
    const bool legacy_accepted = legacy_error == ActionError::None;

    std::lock_guard<std::mutex> lock(g_compare_mutex);
    auto& s = g_compare_stats;
    ++s.actions_seen;
    if (!from_mask) ++s.single_checks;
    s.last_native_address = native_address;
    if (native.original_called) ++s.original_calls;

    if (!native.available) {
        ++s.native_unavailable;
        return;
    }

    ++s.compared;
    if (legacy_accepted == native.accepted) {
        ++s.matched;
        ++s.consecutive_matches;
        return;
    }

    ++s.mismatched;
    s.consecutive_matches = 0;
    s.has_last_mismatch = true;
    s.last_slot = slot;
    s.last_row = row;
    s.last_col = col;
    s.last_plant_type = plant_type;
    s.last_legacy_error = static_cast<int>(legacy_error);
    s.last_native_error = static_cast<int>(native.error);
    s.last_planting_reason = native.planting_reason;
    s.last_legacy_accepted = legacy_accepted;
    s.last_native_accepted = native.accepted;

    std::ostringstream msg;
    msg << "[pvz_gym][PlantDualValidation] mismatch"
        << " slot=" << slot
        << " row=" << row
        << " col=" << col
        << " plant_type=" << plant_type
        << " legacy=" << (legacy_accepted ? "LEGAL" : "ILLEGAL")
        << " legacy_error=" << static_cast<int>(legacy_error)
        << " native=" << (native.accepted ? "LEGAL" : "ILLEGAL")
        << " planting_reason=" << native.planting_reason
        << "\n";
    ::OutputDebugStringA(msg.str().c_str());
}

class MainLoopFreezeGuard {
public:
    explicit MainLoopFreezeGuard(PvzRuntime::VersionedPvZ& pvz)
        : pvz_(pvz) {
        if (!pvz_.GameOn()) return;
        if (SimulationClock::can_reuse_headless_barrier_nolock(pvz_) || pvz::runtime::ResidentExecutor::instance().owner()) {
            borrowed_ = true;
            return;
        }
        pvz_.enable_hack(pvz_.data().block_main_loop, true);
        const int frame_ms = std::max(1, pvz_.GetFrameDuration());
        ::Sleep(static_cast<DWORD>(frame_ms * 2));
        active_ = true;
    }

    ~MainLoopFreezeGuard() {
        if (active_) {
            pvz_.enable_hack(pvz_.data().block_main_loop, false);
        }
    }

    bool active() const noexcept { return active_ || borrowed_; }

private:
    PvzRuntime::VersionedPvZ& pvz_;
    bool active_ = false;
    bool borrowed_ = false;
};

OriginalPlantCallResult make_result(ActionError error) {
    return {true, error == ActionError::None, error};
}

int seed_bank_slots_unlocked(PvzRuntime::VersionedPvZ& pvz) {
    const uintptr_t board = pvz.ReadMemory<uintptr_t>({
        pvz.data().lawn, pvz.data().board});
    return runtime::seed_bank_slot_count_nolock(pvz, board);
}

bool native_conveyor_spatial_authority_unlocked(
    PvzRuntime::VersionedPvZ& pvz, uintptr_t board) {
    if (!board || !runtime::has_conveyor_seed_bank_nolock(pvz, board)) {
        return false;
    }
    const int mode = pvz.GameMode();
    if (mode == 0) {
        const int level = pvz.ReadMemory<int>({board + pvz.data().adventure_level});
        return level == 5 || level == 10;
    }
    return mode == 17 || mode == 33; // Wall-nut Bowling challenges.
}

}  // namespace

OriginalPlantCallResult PlantCalls::query_legality(
    int slot, int row, int col) {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) {
        return make_result(ActionError::GameNotFound);
    }

    MainLoopFreezeGuard freeze(pvz);
    const int bank_slots = seed_bank_slots_unlocked(pvz);
    PlantContext ctx;
    const ActionError gate = build_card_context_unlocked(
        pvz, slot, row, col, ctx);
    const ActionError legacy_error = gate == ActionError::None
        ? validate_plant_context_unlocked(pvz, row, col, ctx, nullptr)
        : gate;
    if (slot >= bank_slots && slot < kMaskSlots) {
        return make_result(legacy_error);
    }
    const NativeLegalityResult native =
        query_native_legality_unlocked(
            pvz, slot, row, col, freeze.active());
    record_comparison(
        slot, row, col, native.plant_type,
        legacy_error, native,
        PvzRuntime::can_plant_at_address(pvz), false);

    const uintptr_t board = pvz.ReadMemory<uintptr_t>({
        pvz.data().lawn, pvz.data().board});
    if (native_conveyor_spatial_authority_unlocked(pvz, board)) {
        return make_result(native.available
            ? native.error : ActionError::UnsupportedGameVersion);
    }
    return make_result(legacy_error);
}

bool build_dropped_action_mask_unlocked(
    PvzRuntime::VersionedPvZ& pvz, uintptr_t board,
    const game::CardSources& sources, std::vector<bool>& out_mask) {
    if (!board || pvz.GameUI() != 3) return false;
    int rows = pvz.GetRowCount();
    if (rows <= 0) rows = 5;
    for (int slot = sources.bank_slots; slot < kMaskSlots; ++slot) {
        const auto& card = sources.slots[static_cast<std::size_t>(slot)];
        if (card.source != game::CardSourceKind::DroppedCoin ||
            card.gate != ActionError::None) continue;
        PlantContext source;
        source.board = board;
        source.source = card.source;
        source.plant_type = card.raw_type;
        source.actual_plant_type = card.plant_type;
        source.coin = card.coin;
        for (int row = 0; row < kMaskRows; ++row) {
            for (int col = 0; col < kMaskCols; ++col) {
                if (row >= rows) continue;
                PlantContext ctx = source;
                const int offset =
                    (slot * kMaskRows + row) * kMaskCols + col;
                out_mask[static_cast<std::size_t>(offset)] =
                    validate_plant_context_unlocked(
                        pvz, row, col, ctx, nullptr) == ActionError::None;
            }
        }
    }
    return true;
}

bool PlantCalls::build_flat_action_mask_shadow(
    std::vector<bool>& out_mask) {
    out_mask.assign(kPlantMaskSize, false);

    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return false;

    MainLoopFreezeGuard freeze(pvz);
    const int bank_slots = seed_bank_slots_unlocked(pvz);
    const uintptr_t board = pvz.ReadMemory<uintptr_t>({
        pvz.data().lawn, pvz.data().board});
    if (bank_slots == 0) {
        game::CardSources sources;
        return game::read_card_sources_nolock(pvz, board, 0, sources) &&
            build_dropped_action_mask_unlocked(pvz, board, sources, out_mask);
    }
    const bool native_conveyor_spatial =
        native_conveyor_spatial_authority_unlocked(pvz, board);

    std::array<ActionError, kPlantMaskSize> legacy_errors{};
    for (int slot = 0; slot < bank_slots; ++slot) {
        for (int row = 0; row < kMaskRows; ++row) {
            for (int col = 0; col < kMaskCols; ++col) {
                const int offset =
                    (slot * kMaskRows + row) * kMaskCols + col;
                const ActionError error =
                    validate_unlocked(pvz, slot, row, col, nullptr);
                legacy_errors[static_cast<std::size_t>(offset)] = error;
                out_mask[static_cast<std::size_t>(offset)] =
                    error == ActionError::None;
            }
        }
    }

    NativeGlobalContext global;
    global.error = build_native_global_context_unlocked(pvz, global);
    if (global.error == ActionError::None) {
        for (int slot = 0; slot < bank_slots; ++slot) {
            global.slots[static_cast<std::size_t>(slot)] =
                read_native_slot_gate_unlocked(pvz, global, slot);
        }
    }

    std::array<NativeLegalityResult, kPlantMaskSize> native_results{};
    std::vector<NativePlantQuery> queries;
    queries.reserve(kPlantMaskSize);

    for (int slot = 0; slot < bank_slots; ++slot) {
        for (int row = 0; row < kMaskRows; ++row) {
            for (int col = 0; col < kMaskCols; ++col) {
                const int offset =
                    (slot * kMaskRows + row) * kMaskCols + col;
                auto& result =
                    native_results[static_cast<std::size_t>(offset)];

                if (global.error != ActionError::None) {
                    result.available = true;
                    result.error = global.error;
                    continue;
                }
                if (row >= global.rows) {
                    result.available = true;
                    result.error = ActionError::InvalidGridCoordinates;
                    continue;
                }

                const auto& gate =
                    global.slots[static_cast<std::size_t>(slot)];
                result.plant_type = gate.actual_plant_type;
                if (gate.error != ActionError::None) {
                    result.available = true;
                    result.error = gate.error;
                    continue;
                }

                NativePlantQuery query;
                query.offset = offset;
                query.slot = slot;
                query.row = row;
                query.col = col;
                query.plant_type = gate.actual_plant_type;
                query.board = global.board;
                queries.push_back(query);
            }
        }
    }

    std::vector<int> reasons;
    const bool native_batch_ok =
        execute_native_can_plant_batch_unlocked(
            pvz, queries, reasons, freeze.active());

    if (native_batch_ok && reasons.size() == queries.size()) {
        for (std::size_t i = 0; i < queries.size(); ++i) {
            auto& result =
                native_results[static_cast<std::size_t>(queries[i].offset)];
            result.original_called = true;
            result.plant_type = queries[i].plant_type;
            if (reasons[i] < 0) {
                result.available = false;
                result.error = ActionError::UnsupportedGameVersion;
                continue;
            }
            result.available = true;
            result.planting_reason = reasons[i];
            result.accepted = reasons[i] == 0;
            result.error = result.accepted
                ? ActionError::None
                : ActionError::InvalidGridCoordinates;
        }
    } else {
        for (const auto& query : queries) {
            auto& result =
                native_results[static_cast<std::size_t>(query.offset)];
            result.original_called = true;
            result.plant_type = query.plant_type;
            result.available = false;
            result.error = ActionError::UnsupportedGameVersion;
        }
    }

    {
        std::lock_guard<std::mutex> stats_lock(g_compare_mutex);
        ++g_compare_stats.mask_batches;
        g_compare_stats.last_native_address =
            PvzRuntime::can_plant_at_address(pvz);
    }

    for (int slot = 0; slot < bank_slots; ++slot) {
        for (int row = 0; row < kMaskRows; ++row) {
            for (int col = 0; col < kMaskCols; ++col) {
                const int offset =
                    (slot * kMaskRows + row) * kMaskCols + col;
                const auto& native =
                    native_results[static_cast<std::size_t>(offset)];
                record_comparison(
                    slot, row, col, native.plant_type,
                    legacy_errors[static_cast<std::size_t>(offset)],
                    native, PvzRuntime::can_plant_at_address(pvz), true);
                if (native_conveyor_spatial) {
                    out_mask[static_cast<std::size_t>(offset)] =
                        native.available && native.accepted;
                }
            }
        }
    }

    if (bank_slots < kMaskSlots) {
        game::CardSources sources;
        if (!game::read_card_sources_nolock(
                pvz, board, pvz.ReadMemory<int>({board + pvz.data().sun}),
                sources) ||
            !build_dropped_action_mask_unlocked(pvz, board, sources, out_mask)) {
            return false;
        }
    }

    return true;
}

OriginalPlantCallResult commit_context_unlocked(
    PvzRuntime::VersionedPvZ& pvz, const PlantContext& ctx,
    int slot, int row, int col);

OriginalPlantCallResult PlantCalls::commit(
    int slot, int row, int col) {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) {
        return make_result(ActionError::GameNotFound);
    }

    PlantContext ctx;
    ActionError error = ActionError::Unsupported;
    {
        MainLoopFreezeGuard freeze(pvz);
        const int bank_slots = seed_bank_slots_unlocked(pvz);
        const ActionError gate = build_card_context_unlocked(
            pvz, slot, row, col, ctx);
        const ActionError legacy_error = gate == ActionError::None
            ? validate_plant_context_unlocked(pvz, row, col, ctx, nullptr)
            : gate;
        if (slot >= bank_slots && slot < kMaskSlots) {
            error = legacy_error;
        } else {
            const NativeLegalityResult native =
                query_native_legality_unlocked(
                    pvz, slot, row, col, freeze.active());
            record_comparison(
                slot, row, col, native.plant_type,
                legacy_error, native,
                PvzRuntime::can_plant_at_address(pvz), false);
            const uintptr_t board = pvz.ReadMemory<uintptr_t>({
                pvz.data().lawn, pvz.data().board});
            if (native_conveyor_spatial_authority_unlocked(pvz, board)) {
                error = native.available
                    ? native.error : ActionError::UnsupportedGameVersion;
                if (error == ActionError::None) {
                    // The original board owns the spatial decision. Build a
                    // fresh packet context without imposing the handwritten
                    // occupancy rules on moving bowling nuts.
                    PlantContext native_ctx;
                    error = build_card_context_unlocked(
                        pvz, slot, row, col, native_ctx);
                    if (error == ActionError::None) ctx = native_ctx;
                }
            } else {
                error = legacy_error;
            }
        }
    }
    if (error != ActionError::None) return make_result(error);

    return commit_context_unlocked(pvz, ctx, slot, row, col);
}

OriginalPlantCallResult commit_context_unlocked(
    PvzRuntime::VersionedPvZ& pvz, const PlantContext& ctx,
    int slot, int row, int col) {
    if (ctx.source == game::CardSourceKind::DroppedCoin &&
        (pvz.ReadMemory<bool>({ctx.coin + PvzRuntime::kCoinDeadOffset}) ||
         pvz.ReadMemory<bool>({ctx.coin + PvzRuntime::kCoinBeingCollectedOffset}) ||
         pvz.ReadMemory<int>({ctx.coin + PvzRuntime::kCoinTypeOffset}) !=
             game::kSeedCoinType ||
         pvz.ReadMemory<int>({ctx.coin + PvzRuntime::kCoinPlantTypeOffset}) !=
             ctx.plant_type)) {
        return make_result(ActionError::CardUnavailable);
    }

    int commit_col = col;
    if (ctx.actual_plant_type == 47 && ctx.kernel_left) {
        commit_col = pvz.ReadMemory<int>({
            ctx.kernel_left + pvz.data().plant_col});
    }

    // Bowling nuts can pass through a cell that receives another nut. Verify
    // a newly allocated DataArray ID after PutPlant, not merely an existing
    // plant of the same type at the target cell.
    std::vector<std::uint32_t> existing_ids;
    const uintptr_t before_plants =
        pvz.ReadMemory<uintptr_t>({ctx.board + pvz.data().plant});
    int before_max =
        pvz.ReadMemory<int>({ctx.board + pvz.data().plant_count_max});
    before_max = std::max(0, std::min(1024, before_max));
    if (before_plants) {
        existing_ids.reserve(static_cast<std::size_t>(before_max));
        for (int i = 0; i < before_max; ++i) {
            const uintptr_t p =
                before_plants + static_cast<uintptr_t>(i) * kPlantStructSize;
            if (pvz.ReadMemory<bool>({p + pvz.data().plant_dead})) continue;
            const std::uint32_t id = pvz.ReadMemory<std::uint32_t>({
                p + kPlantStructSize - sizeof(std::uint32_t)});
            if ((id & 0xFFFF0000u) != 0 &&
                (id & 0xFFFFu) == static_cast<std::uint32_t>(i)) {
                existing_ids.push_back(id);
            }
        }
    }

    if (ctx.imitater) {
        put_imitater(pvz, row, commit_col, ctx.actual_plant_type);
    } else if (SimulationClock::can_reuse_headless_barrier_nolock(pvz)) {
        // Reuse the original instruction generator and the verified headless
        // barrier; VersionedPvZ dispatches injection through its bounded wrapper.
        const int mode = pvz.GameMode();
        pvz.asm_init();
        pvz.asm_put_plant(row, commit_col, ctx.actual_plant_type, false,
                          mode >= 61 && mode <= 70);
        pvz.asm_ret();
        pvz.asm_code_inject();
    } else {
        pvz.PutPlant(row, commit_col, ctx.actual_plant_type, false);
    }

    bool planted = false;
    const uintptr_t plants =
        pvz.ReadMemory<uintptr_t>({ctx.board + pvz.data().plant});
    int plant_max =
        pvz.ReadMemory<int>({ctx.board + pvz.data().plant_count_max});
    plant_max = std::max(0, std::min(1024, plant_max));
    if (plants) {
        for (int i = 0; i < plant_max; ++i) {
            const uintptr_t p =
                plants + static_cast<uintptr_t>(i) * kPlantStructSize;
            if (pvz.ReadMemory<bool>({p + pvz.data().plant_dead})) continue;
            const int spawned =
                pvz.ReadMemory<int>({p + pvz.data().plant_type});
            const int spawned_imitater =
                pvz.ReadMemory<int>({p + pvz.data().plant_imitater});
            const bool expected =
                ctx.imitater
                    ? (spawned == 48 &&
                       spawned_imitater == ctx.actual_plant_type)
                    : spawned == ctx.actual_plant_type;
            const std::uint32_t id = pvz.ReadMemory<std::uint32_t>({
                p + kPlantStructSize - sizeof(std::uint32_t)});
            if (pvz.ReadMemory<int>({p + pvz.data().plant_row}) == row &&
                pvz.ReadMemory<int>({p + pvz.data().plant_col}) == commit_col &&
                expected &&
                (id & 0xFFFF0000u) != 0 &&
                (id & 0xFFFFu) == static_cast<std::uint32_t>(i) &&
                std::find(existing_ids.begin(), existing_ids.end(), id) ==
                    existing_ids.end() &&
                p != ctx.first_aid_replaced_plant) {
                planted = true;
                break;
            }
        }
    }
    if (!planted) return make_result(ActionError::ActionTimeout);

    if (ctx.source == game::CardSourceKind::SeedBank) {
        pvz.WriteMemory<int>(
            {ctx.sun - ctx.cost}, {ctx.board + pvz.data().sun});
    }

    if (!ctx.imitater && is_upgrade(ctx.actual_plant_type)) {
        if (ctx.actual_plant_type == 47) {
            delete_plant(pvz, ctx.kernel_left);
            delete_plant(pvz, ctx.kernel_right);
        } else if (ctx.actual_plant_type == 43) {
            delete_plant(pvz, ctx.carrier);
        } else {
            delete_plant(pvz, ctx.main_plant);
        }
    }

    if (ctx.first_aid_replaced_plant) {
        delete_plant(pvz, ctx.first_aid_replaced_plant);
    }

    if (!ctx.imitater &&
        ctx.actual_plant_type == 35 &&
        ctx.main_plant) {
        pvz.WriteMemory<bool>(
            {false}, {ctx.main_plant + kSleepingOffset});
        pvz.WriteMemory<int>(
            {0}, {ctx.main_plant + kWakeUpCounterOffset});
    }

    if (ctx.source == game::CardSourceKind::DroppedCoin) {
        pvz.WriteMemory<bool>(
            {true}, {ctx.coin + PvzRuntime::kCoinDeadOffset});
    } else if (ctx.source == game::CardSourceKind::Conveyor) {
        consume_conveyor_seed_unlocked(pvz, ctx.seed_bank, slot);
    } else {
        const uintptr_t slot_base = static_cast<uintptr_t>(slot) * 0x50;
        pvz.WriteMemory<int>(
            {ctx.cooldown},
            {ctx.seed_bank + pvz.data().slot_seed_cd_total + slot_base});
        pvz.WriteMemory<int>(
            {0},
            {ctx.seed_bank + pvz.data().slot_seed_cd_past + slot_base});
        pvz.WriteMemory<bool>(
            {false}, {ctx.seed_bank + 0x70 + slot_base});
        pvz.WriteMemory<bool>(
            {true}, {ctx.seed_bank + 0x71 + slot_base});
    }

    return make_result(ActionError::None);
}

std::string PlantCalls::comparison_stats_json() {
    std::lock_guard<std::mutex> lock(g_compare_mutex);
    const auto& s = g_compare_stats;

    std::ostringstream ss;
    ss << "{";
    ss << "\"mode\":\"dual_shadow\",";
    ss << "\"authoritative\":\"native_spatial_for_1_5_1_10_bowling_else_legacy\",";
    ss << "\"native_spatial\":\"Board::CanPlantAt\",";
    ss << "\"actions_seen\":" << s.actions_seen << ",";
    ss << "\"compared\":" << s.compared << ",";
    ss << "\"matched\":" << s.matched << ",";
    ss << "\"mismatched\":" << s.mismatched << ",";
    ss << "\"native_unavailable\":" << s.native_unavailable << ",";
    ss << "\"original_calls\":" << s.original_calls << ",";
    ss << "\"mask_batches\":" << s.mask_batches << ",";
    ss << "\"single_checks\":" << s.single_checks << ",";
    ss << "\"consecutive_matches\":" << s.consecutive_matches << ",";
    ss << "\"last_native_address\":" << s.last_native_address << ",";
    ss << "\"last_mismatch\":";
    if (!s.has_last_mismatch) {
        ss << "null";
    } else {
        ss << "{";
        ss << "\"slot\":" << s.last_slot << ",";
        ss << "\"row\":" << s.last_row << ",";
        ss << "\"col\":" << s.last_col << ",";
        ss << "\"plant_type\":" << s.last_plant_type << ",";
        ss << "\"legacy_error\":" << s.last_legacy_error << ",";
        ss << "\"native_error\":" << s.last_native_error << ",";
        ss << "\"planting_reason\":" << s.last_planting_reason << ",";
        ss << "\"legacy_accepted\":"
           << (s.last_legacy_accepted ? "true" : "false") << ",";
        ss << "\"native_accepted\":"
           << (s.last_native_accepted ? "true" : "false");
        ss << "}";
    }
    ss << "}";
    return ss.str();
}

void PlantCalls::reset_comparison_stats() {
    std::lock_guard<std::mutex> lock(g_compare_mutex);
    g_compare_stats = CompareStats{};
}

}  // namespace pvz::game::original
