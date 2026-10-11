#include "../src/process/thread_safe_point.h"
#include "test_check.h"

#include <cstring>

namespace {

struct SpinThread {
    unsigned char* code;
    HANDLE thread;

    SpinThread() {
        const unsigned char stub[] = {0xeb, 0xfe, 0x31, 0xc0, 0xc2, 0x04, 0x00};
        code = static_cast<unsigned char*>(::VirtualAlloc(
            nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
        CHECK(code);
        std::memcpy(code, stub, sizeof(stub));
        ::FlushInstructionCache(::GetCurrentProcess(), code, sizeof(stub));
        thread = ::CreateThread(nullptr, 0,
            reinterpret_cast<LPTHREAD_START_ROUTINE>(code), nullptr, 0, nullptr);
        CHECK(thread);
        ::Sleep(10);
    }

    ~SpinThread() {
        // Stop without terminating a thread or leaving a suspension held.
        CHECK(::SuspendThread(thread) == 0);
        code[1] = 0; // jmp to xor eax,eax; ret 4
        ::FlushInstructionCache(::GetCurrentProcess(), code, 2);
        CHECK(::ResumeThread(thread) == 1);
        CHECK(::WaitForSingleObject(thread, 5000) == WAIT_OBJECT_0);
        ::CloseHandle(thread);
        CHECK(::VirtualFree(code, 0, MEM_RELEASE));
    }
};

void test_acquires_only_the_verified_instruction() {
    using namespace pvz::process;
    SpinThread spin;
    DWORD observed = 0;
    auto status = SafePointStatus::Retry;
    for (int attempt = 0; attempt < 100 && status == SafePointStatus::Retry; ++attempt) {
        status = try_suspend_at_safe_point(spin.thread,
            reinterpret_cast<std::uintptr_t>(spin.code), &observed);
        if (status == SafePointStatus::Retry) ::Sleep(1);
    }
    CHECK(status == SafePointStatus::Acquired);
    CHECK(observed == reinterpret_cast<std::uintptr_t>(spin.code));
    CHECK(::ResumeThread(spin.thread) == 1);

    CHECK(try_suspend_at_safe_point(spin.thread, 1) == SafePointStatus::Retry);
    CHECK(::SuspendThread(spin.thread) == 0); // retry balanced its suspension
    CHECK(try_suspend_at_safe_point(spin.thread,
        reinterpret_cast<std::uintptr_t>(spin.code)) == SafePointStatus::Failed);
    CHECK(::ResumeThread(spin.thread) == 1); // preserved someone else's count
}

void test_failed_context_read_balances_suspend_count() {
    using namespace pvz::process;
    SpinThread spin;
    HANDLE no_context = ::OpenThread(THREAD_SUSPEND_RESUME, FALSE, ::GetThreadId(spin.thread));
    CHECK(no_context);
    CHECK(try_suspend_at_safe_point(no_context,
        reinterpret_cast<std::uintptr_t>(spin.code)) == SafePointStatus::Failed);
    ::CloseHandle(no_context);
    CHECK(::SuspendThread(spin.thread) == 0);
    CHECK(::ResumeThread(spin.thread) == 1);
    CHECK(try_suspend_at_safe_point(nullptr, 1) == SafePointStatus::Failed);
}

} // namespace

int main(int argc, char** argv) {
    static_assert(sizeof(void*) == 4, "The target game and fixture are x86");
    if (argc == 2 && std::strcmp(argv[1], "--spin-fixture") == 0) {
        SpinThread spin;
        std::cout << std::hex << reinterpret_cast<std::uintptr_t>(spin.code)
                  << ' ' << std::dec << ::GetThreadId(spin.thread) << std::endl;
        std::cin.get();
        return 0;
    }
    test_acquires_only_the_verified_instruction();
    test_failed_context_read_balances_suspend_count();
    std::cout << "safe point: exact EIP, retries, foreign suspension, context failure passed\n";
}
