// Execute the production x86 stub using the game's actual calling conventions.
#include "../src/process/level_reset_code.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

struct App {
    int depth = 0;
    int boards = 1;
    int resets = 0;
    int collections = 0;
    bool almanac_open = false;
    int dialog_kills = 0;
    std::vector<int> delete_depths;
};
static App* app;
static HANDLE blocked, entered, finished;
static bool block_reset = false;
static DWORD reset_thread_id = 0;

static void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}

static void __stdcall collect(App* current) {
    require(current == app, "collector app argument");
    ++current->collections;
    for (auto it = current->delete_depths.begin(); it != current->delete_depths.end();) {
        if (current->depth <= *it) {
            --current->boards;
            it = current->delete_depths.erase(it);
        } else ++it;
    }
}

static void __stdcall reset_impl(App* current, int mode, int restart) {
    require(current == app && mode == 0 && restart == 0, "PreNewGame ABI");
    require(!current->almanac_open, "Almanac must close before PreNewGame");
    if (block_reset) {
        reset_thread_id = GetCurrentThreadId();
        SetEvent(entered);
        WaitForSingleObject(blocked, INFINITE);
        SetEvent(finished);
    }
    current->delete_depths.push_back(current->depth);
    ++current->boards;
    ++current->resets;
}

static void __stdcall kill_dialog_impl(App* current, int dialog_id) {
    require(current == app && dialog_id == 3, "KillDialog ABI and Almanac ID");
    ++current->dialog_kills;
    current->almanac_open = false;
}

// LawnApp::KillDialog passes app in ECX and pops one stack argument.
static __declspec(naked) void kill_dialog_entry() {
    __asm {
        push [esp + 4]
        push ecx
        call kill_dialog_impl
        ret 4
    }
}

// Game PreNewGame passes app in ESI and pops the two stack arguments.
static __declspec(naked) void reset_entry() {
    __asm {
        push [esp + 8]
        push [esp + 8]
        push esi
        call reset_impl
        ret 8
    }
}

static void build(Pt::Code& code, bool close_almanac = true) {
    pvz::process::build_level_reset_code(code,
        reinterpret_cast<unsigned int>(&app),
        reinterpret_cast<unsigned int>(&reset_entry),
        reinterpret_cast<unsigned int>(&collect),
        close_almanac ? reinterpret_cast<unsigned int>(&kill_dialog_entry) : 0);
}

int main() {
    App state;
    app = &state;
    // Reproduce widgets left at depth zero: depth-one normal updates skip them.
    state.delete_depths.push_back(0);
    ++state.boards;
    state.depth = 1;
    collect(app);
    require(state.boards == 2, "fixture must reproduce deferred deletion");
    state.depth = 0;
    Pt::Code code;
    build(code);
    for (int i = 0; i < 2000; ++i) {
        // Both an open Almanac and an absent Dialog must allow a fresh Board.
        state.almanac_open = (i % 2 == 0);
        DWORD exit = STILL_ACTIVE;
        require(code.asm_code_inject(GetCurrentProcess(), 2000, &exit), "reset completion");
        require(exit == 0 && state.boards == 1 && state.delete_depths.empty(),
                "reset accumulated deferred Boards");
    }
    require(state.resets == 2000 && state.collections == 4001, "collector before/after reset");
    require(state.dialog_kills == 2000, "close Almanac on every native reset");

    // Versions without a verified KillDialog entry preserve the existing stub.
    build(code, false);
    require(code.asm_code_inject(GetCurrentProcess(), 2000), "reset without dialog entry");
    require(state.dialog_kills == 2000 && state.resets == 2001 && state.boards == 1,
            "unsupported dialog entry must not be called");

    code.asm_init();
    code.asm_mov_exx(Pt::Reg::EAX, 7);
    code.asm_add_list(0xC2, 0x04, 0x00);
    DWORD failure_exit = STILL_ACTIVE;
    require(code.asm_code_inject(GetCurrentProcess(), 2000, &failure_exit) && failure_exit == 7,
            "propagate unsuccessful remote exit");

    blocked = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    entered = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    finished = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    require(blocked && entered && finished, "timeout events");
    block_reset = true;
    build(code);
    DWORD exit = STILL_ACTIVE;
    ULONGLONG start = GetTickCount64();
    require(!code.asm_code_inject(GetCurrentProcess(), 100, &exit), "blocked reset must time out");
    require(GetTickCount64() - start < 2000 && exit == STILL_ACTIVE, "bounded wait");
    require(WaitForSingleObject(entered, 1000) == WAIT_OBJECT_0, "remote thread entered reset");
    HANDLE remote = OpenThread(SYNCHRONIZE, FALSE, reset_thread_id);
    require(remote != nullptr, "observe timed-out thread");
    require(Pt::Code::has_pending_remote_call(GetCurrentProcess()), "pending remote call");
    require(!code.asm_code_inject(GetCurrentProcess(), 100) && GetLastError() == ERROR_BUSY, "reject overlapping injection");
    void* output = VirtualAllocEx(GetCurrentProcess(), nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    require(output != nullptr, "allocate output");
    Pt::Code::release_remote_allocation(GetCurrentProcess(), output);
    MEMORY_BASIC_INFORMATION info{};
    require(VirtualQuery(output, &info, sizeof(info)) && info.State == MEM_COMMIT, "retain pending output");
    // Returning from timeout must not free code that the remote thread resumes.
    SetEvent(blocked);
    require(WaitForSingleObject(finished, 2000) == WAIT_OBJECT_0, "thread can resume after timeout");
    require(WaitForSingleObject(remote, 2000) == WAIT_OBJECT_0, "remote stub completed");
    CloseHandle(remote);
    require(Pt::Code::reap_remote_calls() == 1, "reap completed timed-out call");
    require(VirtualQuery(output, &info, sizeof(info)) && info.State == MEM_FREE, "free output after remote completion");
    require(!Pt::Code::has_pending_remote_call(GetCurrentProcess()), "clear pending state");
    block_reset = false;
    require(code.asm_code_inject(GetCurrentProcess(), 2000), "injection recovers after reaping");
    require(state.resets == 2003 && state.boards == 1, "stub survived timeout");
    CloseHandle(blocked); CloseHandle(entered); CloseHandle(finished);
    std::puts("2000 x86 resets: Almanac closes before Board replacement; bounded live Boards; timeout resumes safely");
}
