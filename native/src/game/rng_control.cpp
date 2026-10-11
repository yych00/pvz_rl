#include "rng_control.h"

#include "../process/process_manager.h"
#include "runtime/pvz_runtime_layout.h"

#include <limits>
#include <mutex>

namespace pvz::game {
namespace {

// Verified for the target PvZ 1.2.0.1065/GOTY layout. LawnApp::mAppRandSeed is
// copied into Board::mBoardRandSeed while the new Board is constructed, so it
// must be written before LawnApp::PreNewGame creates the episode Board.
constexpr uintptr_t kGotyAppRandSeedOffset = 0x870;

bool supported_version(PvzRuntime::VersionedPvZ& pvz) noexcept {
    return pvz.version_id() == PVZ_1_2_0_1065_EN;
}

}  // namespace

RngSeedSupport RngControl::support() {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return RngSeedSupport::Unsupported;
    if (!pvz.GameOn()) return RngSeedSupport::Unsupported;
    return supported_version(pvz) ? RngSeedSupport::Full : RngSeedSupport::Unsupported;
}

bool RngControl::set_episode_seed(std::uint64_t seed) {
    if (seed > std::numeric_limits<std::uint32_t>::max()) return false;

    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return false;
    if (!pvz.GameOn() || !supported_version(pvz)) return false;

    const uintptr_t lawn = pvz.ReadMemory<uintptr_t>({pvz.data().lawn});
    if (!lawn) return false;

    const std::uint32_t value = static_cast<std::uint32_t>(seed);
    pvz.WriteMemory<std::uint32_t>(value, {pvz.data().lawn, kGotyAppRandSeedOffset});
    return pvz.ReadMemory<std::uint32_t>({pvz.data().lawn, kGotyAppRandSeedOffset}) == value;
}

}  // namespace pvz::game
