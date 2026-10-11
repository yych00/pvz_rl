#include "../src/game/board_update_batch.h"
#include "../src/game/board_update_batch_injector.h"
#include "test_check.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

struct Board { int marker = 0; bool complete = false; };
struct EffectPool { unsigned live = 0; unsigned dead = 0; unsigned recycled = 0; };
struct Lawn { Board* board; int scene; int mode = 0; EffectPool* effects = nullptr; };
enum class Change { None, Scene, BoardNull, BoardReplace, AppNull, InvalidateBoard, Complete };
Board board;
Board replacement;
Lawn lawn;
EffectPool effects;
Lawn* app = &lawn;
Board* expected_board = &board;
int expected_scene = 3;
int updates = 0;
int change_after = 0;
int next_scene = 4;
Change change = Change::None;
int board_deletions = 0;
int effect_deletions = 0;
bool simulate_fixed_pool = false;
std::vector<int> lifecycle;

void __fastcall delete_board(Board* current, void*) {
    CHECK(current == expected_board);
    CHECK(app && app->board == current && app->scene == expected_scene);
    ++board_deletions;
    lifecycle.push_back(1);
}

// Exercise the game's ESI argument/plain-ret convention in executable x86,
// rather than accidentally validating a second __thiscall update instead.
__declspec(naked) void delete_board_esi() {
    __asm {
        mov ecx, esi
        xor edx, edx
        jmp delete_board
    }
}

void __stdcall delete_effects(EffectPool* current) {
    CHECK(app && current == app->effects);
    ++effect_deletions;
    lifecycle.push_back(3);
    current->live -= current->dead;
    current->recycled += current->dead;
    current->dead = 0;
}

void __fastcall update(Board* current, void*) {
    // A second call after scene/identity changes must fail the test.
    CHECK(current == expected_board);
    CHECK(app && app->board == current && app->scene == expected_scene);
    lifecycle.push_back(2);
    ++updates;
    if (simulate_fixed_pool) {
        // A capacity-one holder must recycle each tick, even inside one batch.
        CHECK(effects.live == 0);
        ++effects.live;
        ++effects.dead;
    }
    if (updates != change_after) return;
    switch (change) {
    case Change::Complete: current->complete = true; break;
    case Change::Scene: lawn.scene = next_scene; break;
    case Change::BoardNull: lawn.board = nullptr; break;
    case Change::BoardReplace: lawn.board = &replacement; break;
    case Change::AppNull: app = nullptr; break;
    case Change::InvalidateBoard: {
        lawn.board = nullptr;
        DWORD previous = 0;
        CHECK(::VirtualProtect(current, 4096, PAGE_NOACCESS, &previous));
        break;
    }
    case Change::None: break;
    }
}

void reset(int scene = 3) {
    expected_board = &board;
    expected_scene = scene;
    board.complete = false;
    effects = {};
    lawn = {expected_board, scene, 0, &effects};
    app = &lawn;
    updates = 0;
    change_after = 0;
    change = Change::None;
    board_deletions = effect_deletions = 0;
    simulate_fixed_pool = false;
    lifecycle.clear();
}

DWORD run(int ticks, bool cleanup = true) {
    const auto stub = pvz::game::build_board_update_batch(
        static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(&app)),
        offsetof(Lawn, board), offsetof(Lawn, scene),
        static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(expected_board)),
        static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(&update)),
        ticks, static_cast<std::uint8_t>(expected_scene), offsetof(Board, complete), offsetof(Lawn, mode),
        cleanup ? static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(&delete_board_esi)) : 0,
        cleanup ? static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(&delete_effects)) : 0,
        offsetof(Lawn, effects));
    CHECK(stub.size <= stub.bytes.size());
    const int issued = pvz::game::execute_board_update_batch(::GetCurrentProcess(), stub, ticks);
    CHECK(issued >= 0);
    return static_cast<DWORD>(issued);
}

void test_full_batch_and_entry_guards() {
    reset(); CHECK(run(64) == 64); CHECK(updates == 64);
    CHECK(board_deletions == 64 && effect_deletions == 64);
    reset(); CHECK(run(0) == 0); CHECK(updates == 0);
    reset(); app = nullptr; CHECK(run(64) == 0);
    reset(); lawn.board = nullptr; CHECK(run(64) == 0);
    reset(); lawn.board = &replacement; CHECK(run(64) == 0);
    reset(); lawn.scene = 4; CHECK(run(64) == 0);
}

void test_native_delete_queue_order_and_pool_reuse() {
    reset(); simulate_fixed_pool = true;
    CHECK(run(64) == 64);
    CHECK(effects.live == 0 && effects.dead == 0 && effects.recycled == 64);
    CHECK(lifecycle.size() == 64 * 3);
    for (std::size_t i = 0; i < lifecycle.size(); i += 3) {
        CHECK(lifecycle[i] == 1 && lifecycle[i + 1] == 2 && lifecycle[i + 2] == 3);
    }
    reset(); CHECK(run(64, false) == 64); // Existing direct builder calls remain compatible.
    CHECK(board_deletions == 0 && effect_deletions == 0);
    reset(); lawn.effects = nullptr; CHECK(run(64) == 64);
    CHECK(board_deletions == 64 && effect_deletions == 0);
    reset(); change = Change::AppNull; change_after = 1;
    CHECK(run(64) == 1); CHECK(board_deletions == 1 && effect_deletions == 0);
    reset(); change = Change::BoardNull; change_after = 1;
    CHECK(run(64) == 1); CHECK(board_deletions == 1 && effect_deletions == 1);
}

void test_each_terminal_boundary() {
    for (int scene : {4, 5}) {
        for (int tick = 1; tick <= 64; ++tick) {
            reset(); change = Change::Scene; next_scene = scene; change_after = tick;
            CHECK(run(64) == static_cast<DWORD>(tick));
            CHECK(updates == tick);
        }
    }
    reset(2); change = Change::Scene; next_scene = 3; change_after = 7;
    CHECK(run(64) == 7); CHECK(updates == 7);
    for (int tick = 1; tick <= 64; ++tick) {
        reset(); change = Change::Complete; change_after = tick;
        CHECK(run(64) == static_cast<DWORD>(tick)); CHECK(updates == tick);
    }
    reset(); board.complete = true; CHECK(run(64) == 0);
    reset(); lawn.mode = 1; change = Change::Complete; change_after = 7;
    CHECK(run(64) == 64); // Survival/challenge reuse this flag between stages.
}

void test_board_lifetime_changes() {
    for (auto scenario : {Change::BoardNull, Change::BoardReplace, Change::AppNull}) {
        reset(); change = scenario; change_after = 7;
        CHECK(run(64) == 7); CHECK(updates == 7);
    }
    reset();
    auto* allocation = ::VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    CHECK(allocation);
    expected_board = static_cast<Board*>(allocation);
    lawn.board = expected_board;
    change = Change::InvalidateBoard; change_after = 1;
    CHECK(run(64) == 1); CHECK(updates == 1);
    CHECK(board_deletions == 1 && effect_deletions == 1);
    CHECK(::VirtualFree(allocation, 0, MEM_RELEASE));
}

} // namespace

int main() {
    static_assert(sizeof(void*) == 4, "Execute the actual x86 stub in a Win32 test process");
    test_full_batch_and_entry_guards();
    test_native_delete_queue_order_and_pool_reuse();
    test_each_terminal_boundary();
    test_board_lifetime_changes();
    std::cout << "x86 batch: full 64 ticks, native delete queue order/pool reuse, every WIN/LOSS boundary, intro, null/replaced/freed Board passed\n";
}
