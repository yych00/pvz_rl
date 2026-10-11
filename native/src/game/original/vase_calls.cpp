#include "vase_calls.h"

#include "../../process/process_manager.h"
#include "../../process/main_loop_barrier.h"
#include "../../runtime/resident_executor.h"
#include "../simulation_clock.h"
#include "runtime/conveyor_seed_bank.h"

#include <cstring>
#include <memory>

namespace pvz::game::original {
namespace {
// Verified on retail 1.2.0.1065 (PE timestamp 0x4a37d6af). These are not
// guessed offsets from the Beta build or from another retail version.
constexpr uintptr_t kMouseDown = 0x00411f10;
constexpr uintptr_t kMouseHitTest = 0x00411720;
constexpr uintptr_t kMalletPot = 0x004297c0;
constexpr uintptr_t kOpenPot = 0x00429b20;
constexpr uintptr_t kPotUpdate = 0x00429750;
constexpr int kScaryPotObject = 19;
constexpr int kMalleting = 9;
constexpr std::size_t kItemSize = 0xec;

struct Hit { std::uint32_t object = 0; std::int32_t type = 0; };
struct Pot { uintptr_t address = 0; std::uint32_t identity = 0; int row = -1; int col = -1; };
struct Context { uintptr_t board = 0; uintptr_t challenge = 0; int rows = 0; };

OriginalVaseCallResult result(ActionError error, std::uint32_t identity = 0, uintptr_t board = 0) {
    return {error != ActionError::Unsupported && error != ActionError::UnsupportedGameVersion,
            error == ActionError::None, error, identity, board};
}

template <std::size_t N>
bool signature(PvzRuntime::VersionedPvZ& pvz, uintptr_t address,
               const std::uint8_t (&expected)[N]) {
    std::uint8_t bytes[N]{};
    return pvz.read_remote_bytes(address, bytes, N) && std::memcmp(bytes, expected, N) == 0;
}

bool supported(PvzRuntime::VersionedPvZ& pvz) {
    if (pvz.version_id() != PVZ_1_2_0_1065_EN) return false;
    constexpr std::uint8_t down[] = {0x55, 0x8b, 0xec, 0x83, 0xe4, 0xf8, 0x8b, 0x45, 0x0c};
    constexpr std::uint8_t hit[] = {0x83, 0xec, 0x14, 0x53, 0x55, 0x8b, 0x6c, 0x24, 0x20};
    constexpr std::uint8_t mallet[] = {0x83, 0xec, 0x08, 0x53, 0x8b, 0xd8, 0x8b, 0x43, 0x10};
    constexpr std::uint8_t open[] = {0x64, 0xa1, 0x00, 0x00, 0x00, 0x00, 0x6a, 0xff};
    constexpr std::uint8_t update[] = {0x83, 0x7e, 0x54, 0x09, 0x75, 0x63};
    return signature(pvz, kMouseDown, down) && signature(pvz, kMouseHitTest, hit) &&
           signature(pvz, kMalletPot, mallet) && signature(pvz, kOpenPot, open) &&
           signature(pvz, kPotUpdate, update);
}

class Freeze {
public:
    explicit Freeze(PvzRuntime::VersionedPvZ& pvz) {
        auto& executor = runtime::ResidentExecutor::instance();
        if ((executor.owns(pvz) && executor.owner()) ||
            SimulationClock::can_reuse_headless_barrier_nolock(pvz)) return;
        barrier_ = std::make_unique<process::MainLoopBarrier>(pvz);
        if (!barrier_->acquire())
            throw PvzRuntime::NativeOperationError(ActionError::ActionTimeout, "vase barrier unavailable");
    }
private:
    std::unique_ptr<process::MainLoopBarrier> barrier_;
};

ActionError context(PvzRuntime::VersionedPvZ& pvz, Context& ctx) {
    if (!pvz.GameOn()) return ActionError::GameNotFound;
    if (pvz.GameUI() != 3) return ActionError::GameNotInBattleState;
    ctx.board = pvz.read_memory_checked<uintptr_t>({pvz.data().lawn, pvz.data().board});
    if (!ctx.board) return ActionError::GameNotInBattleState;
    // INTERACT_CELL currently opens vases only. Whack-a-Zombie / hammer
    // targeting is intentionally not implemented; other modes stay masked.
    if (!runtime::is_scary_potter_stage_nolock(pvz, ctx.board)) return ActionError::Unsupported;
    if (!supported(pvz)) return ActionError::UnsupportedGameVersion;
    ctx.rows = pvz.GetRowCount();
    if (ctx.rows != 5 || pvz.GetScene() != 1) return ActionError::Unsupported;
    ctx.challenge = pvz.read_memory_checked<uintptr_t>({ctx.board + 0x160});
    if (!ctx.challenge) return ActionError::GameNotInBattleState;
    if (pvz.read_memory_checked<int>({ctx.challenge + 0x54}) == kMalleting ||
        pvz.read_memory_checked<int>({ctx.board + 0x5748}) > 0) return ActionError::CooldownActive;
    return ActionError::None;
}

std::vector<Pot> pots(PvzRuntime::VersionedPvZ& pvz, const Context& ctx) {
    const auto array = pvz.read_memory_checked<uintptr_t>({ctx.board + pvz.data().grid_item});
    const int count = pvz.read_memory_checked<int>({ctx.board + pvz.data().grid_item_count_max});
    if (count < 0 || count > 128 || (count && !array))
        throw PvzRuntime::NativeOperationError(ActionError::Unsupported, "invalid vase array");
    std::vector<Pot> out;
    for (int index = 0; index < count; ++index) {
        const auto address = array + static_cast<uintptr_t>(index) * kItemSize;
        const auto identity = pvz.read_memory_checked<std::uint32_t>({address + kItemSize - 4});
        if (!(identity & 0xffff0000u) || (identity & 0xffffu) != static_cast<unsigned>(index) ||
            pvz.read_memory_checked<bool>({address + 0x20}) ||
            pvz.read_memory_checked<int>({address + 0x08}) != 7) continue;
        const int col = pvz.read_memory_checked<int>({address + 0x10});
        const int row = pvz.read_memory_checked<int>({address + 0x14});
        if (row >= 0 && row < ctx.rows && col >= 0 && col < 9)
            out.push_back({address, identity, row, col});
    }
    return out;
}

// Read-only original hit tests in one injection: this includes cursor, Dave,
// dialog, fade-out and coin/plant occlusion gates, not just vase occupancy.
std::vector<Hit> hit_tests(PvzRuntime::VersionedPvZ& pvz, const Context& ctx,
                         const std::vector<Pot>& candidates) {
    std::vector<Hit> hits(candidates.size());
    if (hits.empty()) return hits;
    const auto bytes = hits.size() * sizeof(Hit);
    void* remote = ::VirtualAllocEx(pvz.process_handle(), nullptr, bytes,
                                   MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!remote) throw PvzRuntime::NativeOperationError(ActionError::Unsupported, "vase hit allocation failed");
    PvzRuntime::RemoteAllocation allocation(pvz.process_handle(), remote);
    SIZE_T written = 0;
    if (!::WriteProcessMemory(pvz.process_handle(), remote, hits.data(), bytes, &written) || written != bytes)
        throw PvzRuntime::NativeOperationError(ActionError::Unsupported, "vase hit initialization failed");
    pvz.asm_init();
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const auto& pot = candidates[i];
        // Board-local night-lawn cell centre: x=80+80*col, y=130+100*row.
        // MouseHitTest ABI: stack(Board*, x, y, HitResult*), callee ret 16.
        pvz.asm_push_dword(static_cast<unsigned>(reinterpret_cast<uintptr_t>(remote) + i * sizeof(Hit)));
        pvz.asm_push_dword(static_cast<unsigned>(130 + 100 * pot.row));
        pvz.asm_push_dword(static_cast<unsigned>(80 + 80 * pot.col));
        pvz.asm_push_dword(static_cast<unsigned>(ctx.board));
        pvz.asm_call(kMouseHitTest);
    }
    // The original bool result only defines AL. Do not expose stale EAX high
    // bits as the remote thread exit status (which can resemble an NTSTATUS).
    pvz.asm_add_list(0x31, 0xc0);
    pvz.asm_ret();
    pvz.asm_code_inject(true);
    if (!pvz.read_remote_bytes(reinterpret_cast<uintptr_t>(remote), hits.data(), bytes))
        throw PvzRuntime::NativeOperationError(ActionError::Unsupported, "vase hit read failed");
    return hits;
}

OriginalVaseCallResult resolve_unlocked(PvzRuntime::VersionedPvZ& pvz, int row, int col,
                                       Context& ctx) {
    auto error = context(pvz, ctx);
    if (error != ActionError::None) return result(error);
    if (row < 0 || row >= ctx.rows || col < 0 || col >= 9)
        return result(ActionError::InvalidGridCoordinates);
    for (const auto& pot : pots(pvz, ctx)) {
        if (pot.row != row || pot.col != col) continue;
        const auto hit = hit_tests(pvz, ctx, {pot}).front();
        if (hit.type == kScaryPotObject && hit.object == pot.address)
            return result(ActionError::None, pot.identity, ctx.board);
    }
    return result(ActionError::InvalidArgument);
}
} // namespace

OriginalVaseCallResult VaseCalls::resolve(int row, int col) {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    pvz.ensure_remote_idle();
    if (!pvz.GameOn()) return result(ActionError::GameNotFound);
    Context ctx;
    const auto gate = context(pvz, ctx);
    if (gate != ActionError::None) return result(gate);
    runtime::CriticalSectionGuard critical;
    Freeze freeze(pvz);
    return resolve_unlocked(pvz, row, col, ctx);
}

bool VaseCalls::build_flat_action_mask(std::vector<bool>& mask) {
    mask.assign(54, false);
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    pvz.ensure_remote_idle();
    if (!pvz.GameOn()) return false;
    // Avoid acquiring a LIVE barrier in every unrelated game mode.
    Context ctx;
    if (context(pvz, ctx) != ActionError::None) return true;
    runtime::CriticalSectionGuard critical;
    Freeze freeze(pvz);
    if (context(pvz, ctx) != ActionError::None) return true;
    const auto candidates = pots(pvz, ctx);
    const auto hits = hit_tests(pvz, ctx, candidates);
    for (std::size_t i = 0; i < candidates.size(); ++i)
        if (hits[i].type == kScaryPotObject && hits[i].object == candidates[i].address)
            mask[static_cast<std::size_t>(candidates[i].row * 9 + candidates[i].col)] = true;
    return true;
}

OriginalVaseCallResult VaseCalls::commit(int row, int col, std::uint32_t expected_identity,
                                       uintptr_t expected_board) {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    pvz.ensure_remote_idle();
    if (!pvz.GameOn()) return result(ActionError::GameNotFound);
    Context ctx;
    const auto gate = context(pvz, ctx);
    if (gate != ActionError::None) return result(gate);
    runtime::CriticalSectionGuard critical;
    Freeze freeze(pvz);
    const auto resolved = resolve_unlocked(pvz, row, col, ctx);
    if (!resolved.accepted) return resolved;
    if (!expected_board || ctx.board != expected_board)
        return result(ActionError::GameNotInBattleState);
    if (!expected_identity || resolved.identity != expected_identity) return result(ActionError::InvalidArgument);
    // Original Board::MouseDown: ECX=Board*, stack(x,y,positive click count),
    // ret 12. It dispatches to Challenge::ScaryPotterMalletPot; Board::Update
    // subsequently advances the animation and calls ScaryPotterOpenPot.
    // No window messages, wall-clock waiting, forced pot death or manual loot.
    pvz.asm_init();
    pvz.asm_push_dword(1);
    pvz.asm_push_dword(static_cast<unsigned>(130 + 100 * row));
    pvz.asm_push_dword(static_cast<unsigned>(80 + 80 * col));
    pvz.asm_mov_exx(Pt::Reg::ECX, static_cast<unsigned>(ctx.board));
    pvz.asm_call(kMouseDown);
    pvz.asm_add_list(0x31, 0xc0); // MouseDown returns void, not a thread status.
    pvz.asm_ret();
    pvz.asm_code_inject(true);
    if (pvz.read_memory_checked<int>({ctx.challenge + 0x54}) != kMalleting ||
        pvz.read_memory_checked<int>({ctx.challenge + 0xa8}) != col ||
        pvz.read_memory_checked<int>({ctx.challenge + 0xac}) != row)
        return result(ActionError::Unsupported);
    return result(ActionError::None, resolved.identity, ctx.board);
}

} // namespace pvz::game::original
