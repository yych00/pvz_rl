#include "board_update_batch.h"
#include "board_update_batch_injector.h"
#include "runtime/pvz_runtime_layout.h"
#include "../runtime/resident_executor.h"
#include <cstring>

namespace pvz::game {

int inject_board_update_batch_nolock(
    PvzRuntime::VersionedPvZ& pvz, std::uintptr_t board,
    int ticks, std::uintptr_t update_fn, std::uint8_t scene) {
    if (ticks <= 0 || !board || !update_fn) return -1;
    const auto board_delete = PvzRuntime::board_delete_queue_address(pvz);
    const auto effect_delete = PvzRuntime::effect_delete_queue_address(pvz);
    // These two new native calls have different ABIs. Validate both entry
    // signatures before executing a 1065 mutation; no guessed-version fallback.
    // Other versions retain their existing caller-supplied update path without
    // new deletion calls. Their lifecycle stability is not verified here.
    constexpr std::uint8_t board_signature[] = {0x53, 0x55, 0x33, 0xdb, 0x33, 0xc0, 0x57, 0x83, 0xcd, 0xff};
    constexpr std::uint8_t effect_signature[] = {0x53, 0x55, 0x56, 0x33, 0xdb, 0x57, 0x33, 0xf6, 0x3b, 0xf3};
    std::uint8_t observed[sizeof(board_signature)]{};
    if (pvz.version_id() == PVZ_1_2_0_1065_EN && (!board_delete || !effect_delete ||
        !pvz.read_remote_bytes(board_delete, observed, sizeof(observed)) ||
        std::memcmp(observed, board_signature, sizeof(observed)) != 0 ||
        !pvz.read_remote_bytes(effect_delete, observed, sizeof(observed)) ||
        std::memcmp(observed, effect_signature, sizeof(observed)) != 0)) return -1;
    const auto stub = build_board_update_batch(
        static_cast<std::uint32_t>(pvz.data().lawn),
        static_cast<std::uint32_t>(pvz.data().board),
        static_cast<std::uint32_t>(pvz.data().game_ui),
        static_cast<std::uint32_t>(board),
        static_cast<std::uint32_t>(update_fn),
        static_cast<std::uint32_t>(ticks), scene,
        static_cast<std::uint32_t>(pvz.data().sun - 0x5560 + 0x55fc),
        static_cast<std::uint32_t>(pvz.data().game_mode),
        static_cast<std::uint32_t>(board_delete),
        static_cast<std::uint32_t>(effect_delete), 0x820);
    auto& executor = runtime::ResidentExecutor::instance();
    if (executor.owns(pvz)) {
        std::uint32_t issued = 0;
        if (!executor.execute(stub.bytes.data(), stub.size, {}, issued) || issued > static_cast<unsigned>(ticks)) return -1;
        return static_cast<int>(issued);
    }
    return execute_board_update_batch(pvz.process_handle(), stub,
                                     static_cast<std::uint32_t>(ticks));
}

} // namespace pvz::game
