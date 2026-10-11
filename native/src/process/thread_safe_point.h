#pragma once

#include <Windows.h>
#include <cstdint>

namespace pvz::process {

enum class SafePointStatus { Acquired, Retry, Failed };

// The game is x86 even when the host DLL is x64. A successful result leaves
// exactly our suspension held; every other result balances our suspend count.
inline SafePointStatus try_suspend_at_safe_point(
    HANDLE thread, std::uintptr_t instruction, DWORD* observed_ip = nullptr) noexcept {
    const DWORD previous = ::SuspendThread(thread);
    if (previous == static_cast<DWORD>(-1)) return SafePointStatus::Failed;
    if (previous != 0) {
        ::ResumeThread(thread);
        return SafePointStatus::Failed;
    }
#if defined(_WIN64)
    WOW64_CONTEXT context{};
    context.ContextFlags = WOW64_CONTEXT_CONTROL;
    const bool read = ::Wow64GetThreadContext(thread, &context) != FALSE;
#else
    CONTEXT context{};
    context.ContextFlags = CONTEXT_CONTROL;
    const bool read = ::GetThreadContext(thread, &context) != FALSE;
#endif
    if (read && observed_ip) *observed_ip = context.Eip;
    if (read && context.Eip == instruction) return SafePointStatus::Acquired;
    ::ResumeThread(thread);
    return read ? SafePointStatus::Retry : SafePointStatus::Failed;
}

} // namespace pvz::process
