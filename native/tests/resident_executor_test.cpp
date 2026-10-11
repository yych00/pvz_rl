#include "../src/runtime/resident_code.h"
#include "test_check.h"
#include <Windows.h>
#include <cstring>

void exercise(bool event_wait) {
    using namespace pvz::runtime;
    auto* page = static_cast<std::uint8_t*>(::VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
    CHECK(page);
    const auto base = static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(page));
    auto* box = reinterpret_cast<volatile ResidentMailbox*>(page);
    ResidentMailbox initial; initial.code = base + 2048; initial.deadline = ::GetTickCount()+10000;
    std::memcpy(page, &initial, sizeof(initial));
    ResidentImports api; api.sleep = reinterpret_cast<std::uint32_t>(&::Sleep); api.clock = reinterpret_cast<std::uint32_t>(&::GetTickCount);
    HANDLE wake = event_wait ? ::CreateEventW(nullptr,FALSE,FALSE,nullptr) : nullptr;
    if (event_wait) { CHECK(wake); api.wait = reinterpret_cast<std::uint32_t>(&::WaitForSingleObject); }
    const auto stub = build_resident_loop(base, base+64, base+128, 0, api, false,
        static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(wake)));
    std::memcpy(page+256, stub.data(), stub.size());
    HANDLE thread = ::CreateThread(nullptr, 0, reinterpret_cast<LPTHREAD_START_ROUTINE>(page+256), nullptr, 0, nullptr);
    CHECK(thread);
    auto wait = [&](unsigned int expected) {
        DWORD deadline = ::GetTickCount()+5000;
        while (box->acknowledgement != expected && static_cast<int>(::GetTickCount()-deadline) < 0) ::SwitchToThread();
        CHECK(box->acknowledgement == expected);
    };
    wait(1);
    ::Sleep(30); // Wake a genuinely idle resident, not just its hot polling path.
    for (unsigned int i = 1; i <= 1000; ++i) {
        std::uint8_t code[] = {0xb8, 0, 0, 0, 0, 0xc3, 0x04, 0};
        std::memcpy(code+1, &i, 4); if (i%2) code[5] = 0xc2;
        std::memcpy(page+2048, code, sizeof(code)); ::FlushInstructionCache(::GetCurrentProcess(), page+2048, sizeof(code));
        box->acknowledgement = 0; box->command = 2;
        if (wake) CHECK(::SetEvent(wake));
        wait(2);
        CHECK(box->result == i);
        // Hold acknowledgement proves it has left the previous completion
        // wait, so a rapidly following command cannot be missed or run twice.
        box->command = 1; if (wake) CHECK(::SetEvent(wake)); wait(1);
    }
    // Simulate the controller disappearing while a completed result is unread.
    box->acknowledgement = 0; box->command = 2; if (wake) CHECK(::SetEvent(wake)); wait(2);
    box->deadline = ::GetTickCount()+20;
    CHECK(::WaitForSingleObject(thread, 2000) == WAIT_OBJECT_0);
    CHECK(box->acknowledgement == 3);
    ::CloseHandle(thread); ::VirtualFree(page, 0, MEM_RELEASE);
    if (wake) ::CloseHandle(wake);
}

int main() { exercise(false); exercise(true); }
