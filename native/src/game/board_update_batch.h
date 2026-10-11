#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace PvzRuntime { class VersionedPvZ; }

namespace pvz::game {

// Caller owns the verified main-loop barrier and ProcessManager mutex.
// Returns issued updates, or -1 on injection/thread failure.
int inject_board_update_batch_nolock(
    PvzRuntime::VersionedPvZ& pvz, std::uintptr_t board,
    int ticks, std::uintptr_t update_fn, std::uint8_t scene);

struct BoardUpdateBatchCode {
    std::array<std::uint8_t, 128> bytes{};
    std::size_t size = 0;
};

// x86 thread procedure: a single injection still executes the whole batch.
// All lifetime checks run in the target process, before each update. After an
// update invalidates Board, only the current LawnApp and saved registers are
// accessed. Optional native deletion queues restore the parent frame lifecycle:
// Board queue (ESI, ret) -> Board::Update -> EffectSystem queue (stack, ret 4).
inline BoardUpdateBatchCode build_board_update_batch(
    std::uint32_t lawn_address, std::uint32_t board_offset,
    std::uint32_t scene_offset, std::uint32_t board,
    std::uint32_t update_fn, std::uint32_t ticks, std::uint8_t scene,
    std::uint32_t complete_offset = 0, std::uint32_t mode_offset = 0,
    std::uint32_t board_delete_fn = 0, std::uint32_t effect_delete_fn = 0,
    std::uint32_t effect_offset = 0) noexcept {
    BoardUpdateBatchCode out;
    auto byte = [&](std::uint8_t value) { out.bytes[out.size++] = value; };
    auto dword = [&](std::uint32_t value) {
        for (int shift = 0; shift < 32; shift += 8)
            byte(static_cast<std::uint8_t>(value >> shift));
    };
    auto branch = [&](std::uint8_t opcode) {
        byte(opcode);
        const auto offset = out.size;
        byte(0);
        return offset;
    };
    auto target = [&](std::size_t offset, std::size_t destination) {
        out.bytes[offset] = static_cast<std::uint8_t>(
            static_cast<int>(destination) - static_cast<int>(offset + 1));
    };

    byte(0x53); byte(0x56); byte(0x57); // save EBX, ESI, EDI
    byte(0x31); byte(0xDB);             // xor ebx, ebx (issued count)
    byte(0xBE); dword(board);           // mov esi, original Board
    byte(0xBF); dword(ticks);           // mov edi, remaining
    byte(0x85); byte(0xFF);             // test edi, edi
    const auto empty = branch(0x74);

    const auto loop = out.size;
    byte(0xA1); dword(lawn_address);    // mov eax, [LawnApp global]
    byte(0x85); byte(0xC0);             // test eax, eax
    const auto app_gone = branch(0x74);
    byte(0x39); byte(0xB0); dword(board_offset); // cmp [eax+board], esi
    const auto board_changed = branch(0x75);
    byte(0x83); byte(0xB8); dword(scene_offset); byte(scene);
    const auto scene_changed = branch(0x75);
    std::size_t completed = 0;
    if (scene == 3 && complete_offset && mode_offset) {
        byte(0x83); byte(0xb8); dword(mode_offset); byte(0);
        const auto other_mode = branch(0x75);
        byte(0x80); byte(0xbe); dword(complete_offset); byte(0);
        completed = branch(0x75);
        target(other_mode, out.size);
    }
    if (board_delete_fn) {
        byte(0xB8); dword(board_delete_fn);
        byte(0xFF); byte(0xD0);         // ESI = verified current Board, plain ret
    }
    byte(0x8B); byte(0xCE);             // mov ecx, esi
    byte(0xB8); dword(update_fn);
    byte(0xFF); byte(0xD0);             // call eax (Board::Update)
    byte(0x43);                        // inc ebx
    if (effect_delete_fn && effect_offset) {
        // Board::Update can replace/delete Board or LawnApp. Re-read the app,
        // never dereference ESI afterward, and drain only the current holder.
        byte(0xA1); dword(lawn_address);
        byte(0x85); byte(0xC0);
        const auto no_app = branch(0x74);
        byte(0x8B); byte(0x80); dword(effect_offset);
        byte(0x85); byte(0xC0);
        const auto no_effect = branch(0x74);
        byte(0x50);                    // push EffectSystem*
        byte(0xB8); dword(effect_delete_fn);
        byte(0xFF); byte(0xD0);         // native stdcall, ret 4
        target(no_app, out.size);
        target(no_effect, out.size);
    }
    byte(0x4F);                        // dec edi
    const auto repeat = branch(0x75);
    target(repeat, loop);

    const auto done = out.size;
    for (auto offset : {empty, app_gone, board_changed, scene_changed})
        target(offset, done);
    if (completed) target(completed, done);
    byte(0x8B); byte(0xC3);             // mov eax, ebx
    byte(0x5F); byte(0x5E); byte(0x5B);
    byte(0xC2); byte(0x04); byte(0x00); // stdcall thread procedure: ret 4
    return out;
}

} // namespace pvz::game
