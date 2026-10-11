#include "resident_executor.h"
#include "resident_code.h"
#include "../process/main_loop_barrier.h"
#include "../process/process_manager.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <string>
#include <sstream>
#include <thread>

namespace pvz::runtime {
namespace {
struct State {
    HANDLE process = nullptr, thread = nullptr, wake = nullptr, remote_wake = nullptr;
    std::uint32_t pid = 0, base = 0, code = 0;
    bool owner = false, stopping = false;
    std::atomic<bool> heartbeat_stop{false};
    std::atomic<bool> critical_section{false};
    HANDLE heartbeat = nullptr;
    std::uint64_t commands = 0;
    ~State() {
        heartbeat_stop.store(true);
        if (heartbeat) { ::WaitForSingleObject(heartbeat, 5000); ::CloseHandle(heartbeat); }
        if (process && base) {
            const std::uint32_t stop = 3;
            ::WriteProcessMemory(process, reinterpret_cast<void*>(base), &stop, 4, static_cast<SIZE_T*>(nullptr));
        }
        if (thread) ::CloseHandle(thread);
        if (process) ::CloseHandle(process);
    }
};
State& state() { static State* value = new State; return *value; }
constexpr std::size_t kAllocationSize = 128 * 1024;
constexpr std::size_t kCodeOffset = 4096;

struct WaitFailure {
    bool present = false;
    std::uint32_t expected = 0, observed = 0;
    int timeout_ms = 0;
    DWORD win32_error = 0;
};
// Heartbeat RPCs run on another thread and must not overwrite the caller's failure.
thread_local WaitFailure last_wait_failure;

bool write(std::uint32_t address, const void* data, std::size_t size) {
    SIZE_T count = 0;
    return ::WriteProcessMemory(state().process, reinterpret_cast<void*>(address), data, size, &count) && count == size;
}
bool word(std::uint32_t offset, std::uint32_t value) {
    if (!write(state().base + offset, &value, 4)) return false;
    return offset != 0 || !state().wake || ::SetEvent(state().wake) != FALSE;
}
bool read(std::uint32_t address, void* data, std::size_t size) {
    SIZE_T count = 0;
    return ::ReadProcessMemory(state().process, reinterpret_cast<void*>(address), data, size, &count) && count == size;
}
bool wait_ack(std::uint32_t expected, int timeout = 5000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout);
    std::uint32_t ack = 0;
    do {
        if (!read(state().base + 4, &ack, 4)) {
            last_wait_failure = {true, expected, ack, timeout, ::GetLastError()};
            return false;
        }
        if (ack == expected) return true;
        if (ack == 3) {
            last_wait_failure = {true, expected, ack, timeout, 0};
            return false;
        }
        ::SwitchToThread();
    } while (std::chrono::steady_clock::now() < deadline);
    last_wait_failure = {true, expected, ack, timeout, ERROR_TIMEOUT};
    return false;
}

// Resolve the target's PE32 IAT rather than using host addresses. WOW64 user32
// and kernel32 have different addresses from their 64-bit host counterparts.
ResidentImports imports(PvzRuntime::VersionedPvZ& pvz) {
    ResidentImports result;
    const auto nth = 0x400000u + pvz.ReadMemory<std::uint32_t>({0x40003c});
    if (pvz.ReadMemory<std::uint32_t>({nth}) != 0x4550 ||
        pvz.ReadMemory<std::uint16_t>({nth + 24}) != 0x10b) return result;
    const auto table = 0x400000u + pvz.ReadMemory<std::uint32_t>({nth + 24 + 104});
    for (int descriptor = 0; descriptor < 128; ++descriptor) {
        std::uint32_t entry[5]{};
        if (!pvz.read_remote_bytes(table + descriptor * 20, entry, sizeof(entry)) || !entry[3]) break;
        const auto names = entry[0] ? entry[0] : entry[4];
        for (unsigned int i = 0; i < 4096; ++i) {
            const auto thunk = pvz.ReadMemory<std::uint32_t>({0x400000u + names + i * 4});
            if (!thunk) break;
            if (thunk & 0x80000000) continue;
            char name[96]{};
            if (!pvz.read_remote_bytes(0x400000u + thunk + 2, name, sizeof(name) - 1)) continue;
            const auto address = pvz.ReadMemory<std::uint32_t>({0x400000u + entry[4] + i * 4});
            if (!std::strcmp(name, "Sleep")) result.sleep = address;
            else if (!std::strcmp(name, "WaitForSingleObject")) result.wait = address;
            else if (!std::strcmp(name, "GetTickCount")) result.clock = address;
            else if (!std::strcmp(name, "PeekMessageA")) result.peek = address;
            else if (!std::strcmp(name, "TranslateMessage")) result.translate = address;
            else if (!std::strcmp(name, "DispatchMessageA")) result.dispatch = address;
            else if (!std::strcmp(name, "BeginPaint")) result.begin_paint = address;
            else if (!std::strcmp(name, "EndPaint")) result.end_paint = address;
        }
    }
    return result;
}
DWORD WINAPI heartbeat_proc(void*) {
    auto& heartbeat_state = state();
    while (!heartbeat_state.heartbeat_stop.load()) {
        const std::uint32_t expiry = ::GetTickCount() + 10000;
        write(heartbeat_state.base + 12, &expiry, 4);
        if (heartbeat_state.owner && !heartbeat_state.critical_section.load(std::memory_order_relaxed)) {
            auto& manager = process::ProcessManager::instance();
            std::unique_lock<std::mutex> lock(manager.mutex(), std::try_to_lock);
            if (lock.owns_lock() && !heartbeat_state.heartbeat_stop.load() &&
                !heartbeat_state.critical_section.load(std::memory_order_relaxed)) {
                word(0, 0);
                if (wait_ack(0, 100)) { ::Sleep(1); ResidentExecutor::instance().hold(); }
            }
        }
        for (int i = 0; i < 2 && !heartbeat_state.heartbeat_stop.load(); ++i) ::Sleep(20);
    }
    return 0;
}
} // namespace

ResidentExecutor& ResidentExecutor::instance() { static ResidentExecutor value; return value; }
bool ResidentExecutor::owns(const PvzRuntime::VersionedPvZ& pvz) const noexcept {
    return state().base && !state().stopping && state().pid == pvz.process_id() && pvz.process_handle() &&
        ::WaitForSingleObject(state().process, 0) == WAIT_TIMEOUT;
}
bool ResidentExecutor::owner() const noexcept { return state().base && state().owner && !state().stopping; }
std::uint64_t ResidentExecutor::commands() const noexcept { return state().commands; }

std::string ResidentExecutor::diagnostics_json() const {
    std::ostringstream out;
    out << "{\"barrier_last_instruction\":" << process::MainLoopBarrier::last_instruction()
        << ",\"barrier_retry_count\":" << process::MainLoopBarrier::retry_count()
        << ",\"barrier_failure_count\":" << process::MainLoopBarrier::failure_count();
    if (last_wait_failure.present) {
        out << ",\"wait_failure\":{\"expected_ack\":" << last_wait_failure.expected
            << ",\"observed_ack\":" << last_wait_failure.observed
            << ",\"timeout_ms\":" << last_wait_failure.timeout_ms
            << ",\"win32_error\":" << last_wait_failure.win32_error << "}";
    }
    auto& manager = process::ProcessManager::instance();
    std::unique_lock<std::mutex> lock(manager.mutex(), std::try_to_lock);
    out << ",\"state_available\":" << (lock.owns_lock() ? "true" : "false");
    if (lock.owns_lock()) {
        auto& s = state();
        DWORD exit_code = 0;
        const bool exit_available = s.process && ::GetExitCodeProcess(s.process, &exit_code);
        std::uint32_t mailbox[4]{};
        const bool mailbox_available = s.base && read(s.base, mailbox, sizeof(mailbox));
        out << ",\"process_id\":" << s.pid << ",\"mailbox_address\":" << s.base
            << ",\"owner\":" << (s.owner ? "true" : "false")
            << ",\"stopping\":" << (s.stopping ? "true" : "false")
            << ",\"critical_section\":" << (s.critical_section.load() ? "true" : "false")
            << ",\"commands\":" << s.commands
            << ",\"process_exit_code\":" << (exit_available ? std::to_string(exit_code) : "null")
            << ",\"mailbox_available\":" << (mailbox_available ? "true" : "false");
        if (mailbox_available) out << ",\"command\":" << mailbox[0]
            << ",\"ack\":" << mailbox[1] << ",\"heartbeat_deadline\":" << mailbox[3]
            << ",\"current_tick_count\":" << ::GetTickCount();
    }
    return out.str() + "}";
}

bool ResidentExecutor::begin(PvzRuntime::VersionedPvZ& pvz, bool owner) {
    last_wait_failure = {};
    if (owns(pvz)) return state().owner == owner && hold();
    if (state().base && !stop(pvz)) return false;
    if (!pvz.GameOn()) return false;
    const auto api = imports(pvz);
    if (!api.sleep || !api.clock || (owner && (!api.peek || !api.translate || !api.dispatch ||
        !api.begin_paint || !api.end_paint))) return false;
    HANDLE process = nullptr;
    if (!::DuplicateHandle(::GetCurrentProcess(), pvz.process_handle(), ::GetCurrentProcess(),
        &process, 0, FALSE, DUPLICATE_SAME_ACCESS)) return false;
    void* allocated = ::VirtualAllocEx(process, nullptr, kAllocationSize,
        MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
    if (!allocated || reinterpret_cast<std::uintptr_t>(allocated) > UINT32_MAX) {
        if (allocated) ::VirtualFreeEx(process, allocated, 0, MEM_RELEASE);
        ::CloseHandle(process); return false;
    }
    auto& s = state(); s.process = process; s.base = static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(allocated));
    s.pid = pvz.process_id(); s.owner = owner; s.stopping = false; s.code = s.base + kCodeOffset; s.commands = 0;
    if (api.wait) {
        s.wake = ::CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!s.wake || !::DuplicateHandle(::GetCurrentProcess(), s.wake, process,
            &s.remote_wake, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
            if (s.wake) ::CloseHandle(s.wake);
            s.wake = s.remote_wake = nullptr;
        }
    }
    ResidentMailbox box; box.deadline = ::GetTickCount() + 10000; box.code = s.code;
    const auto loop = build_resident_loop(s.base, s.base + 64, s.base + 128,
        static_cast<std::uint32_t>(pvz.data().block_main_loop.mem_addr - 1), api, owner,
        static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(s.remote_wake)));
    const auto entry = s.base + 256;
    bool started = write(s.base, &box, sizeof(box)) && write(entry, loop.data(), loop.size()) &&
        ::FlushInstructionCache(process, reinterpret_cast<void*>(entry), loop.size());
    if (started && owner) {
        process::MainLoopBarrier barrier(pvz);
        started = barrier.acquire() && barrier.handoff(entry);
    } else if (started) {
        s.thread = ::CreateRemoteThread(process, nullptr, 0,
            reinterpret_cast<LPTHREAD_START_ROUTINE>(entry), nullptr, 0, nullptr);
        started = s.thread != nullptr;
    }
    if (!started) {
        if (s.remote_wake) {
            HANDLE local = nullptr;
            if (::DuplicateHandle(process, s.remote_wake, ::GetCurrentProcess(), &local, 0, FALSE,
                DUPLICATE_SAME_ACCESS | DUPLICATE_CLOSE_SOURCE)) ::CloseHandle(local);
        }
        if (s.wake) ::CloseHandle(s.wake);
        s.wake = s.remote_wake = nullptr;
        ::VirtualFreeEx(process, allocated, 0, MEM_RELEASE);
        ::CloseHandle(process); s.process = nullptr; s.base = s.pid = 0; return false;
    }
    s.heartbeat_stop.store(false);
    s.heartbeat = ::CreateThread(nullptr, 0, heartbeat_proc, nullptr, 0, nullptr);
    if (!s.heartbeat) { stop(pvz); return false; }
    if (!wait_ack(1)) { stop(pvz); return false; }
    return true;
}

bool ResidentExecutor::hold() { return state().base && !state().stopping && word(0, 1) && wait_ack(1); }
bool ResidentExecutor::service(int milliseconds) {
    if (!state().base || state().stopping || !word(0, 4) || !wait_ack(0)) return false;
    ::Sleep(static_cast<DWORD>(std::max(1, milliseconds)));
    return hold();
}

bool ResidentExecutor::execute(const std::uint8_t* bytes, std::size_t size,
    const std::vector<unsigned int>& calls, std::uint32_t& result) {
    last_wait_failure = {};
    auto& s = state();
    if (!s.base || !bytes || !size || size > kAllocationSize - kCodeOffset || !hold()) return false;
    std::vector<std::uint8_t> code(bytes, bytes + size);
    for (auto pos : calls) {
        if (pos + 4 > size) return false;
        std::uint32_t address = 0; std::memcpy(&address, code.data() + pos, 4);
        address -= s.code + pos + 4;
        std::memcpy(code.data() + pos, &address, 4);
    }
    if (!write(s.code, code.data(), code.size()) || !::FlushInstructionCache(s.process,
        reinterpret_cast<void*>(s.code), size) || !word(4, 0) || !word(0, 2) || !wait_ack(2)) return false;
    const bool read_result = read(s.base + 8, &result, 4);
    ++s.commands;
    return hold() && read_result;
}

bool ResidentExecutor::stop(PvzRuntime::VersionedPvZ& pvz) noexcept {
    auto& s = state();
    if (!s.base) return true;
    s.stopping = true;
    s.heartbeat_stop.store(true);
    if (s.heartbeat) { ::WaitForSingleObject(s.heartbeat, 6000); ::CloseHandle(s.heartbeat); s.heartbeat = nullptr; }

    bool safe = (s.process && ::WaitForSingleObject(s.process, 0) == WAIT_OBJECT_0);
    if (!safe && s.process) {
        DWORD exit_code = 0;
        if (::GetExitCodeProcess(s.process, &exit_code) && exit_code != STILL_ACTIVE) {
            safe = true;
        }
    }
    if (!safe && s.pid != 0 && pvz.process_id() != s.pid) {
        // 目标进程已发生替换，旧驻留句柄已失去意义
        safe = true;
    }

    if (!safe && word(0, 3)) {
        if (s.owner) {
            if (wait_ack(3, 5000)) {
                // ack precedes the final pop/ret. Park at the ORIGINAL safe point
                // before unmapping the resident instructions.
                process::MainLoopBarrier barrier(pvz);
                safe = pvz.process_id() == s.pid && barrier.acquire();
                if (safe) ::VirtualFreeEx(s.process, reinterpret_cast<void*>(s.base), 0, MEM_RELEASE);
            }
        } else {
            safe = s.thread && ::WaitForSingleObject(s.thread, 5000) == WAIT_OBJECT_0;
            if (safe) ::VirtualFreeEx(s.process, reinterpret_cast<void*>(s.base), 0, MEM_RELEASE);
        }
    }

    if (!safe && s.process) {
        DWORD exit_code = 0;
        if (::GetExitCodeProcess(s.process, &exit_code) && exit_code != STILL_ACTIVE) {
            safe = true;
        }
    }

    // On timeout preserve the command mailbox, PID and handles as well as its
    // memory. Otherwise the next begin() can allocate a competing owner while
    // the previous loop is still executing. Only verified stop can clear it.
    if (!safe) return false;
    if (s.remote_wake) {
        HANDLE local = nullptr;
        if (::DuplicateHandle(s.process, s.remote_wake, ::GetCurrentProcess(), &local, 0, FALSE,
            DUPLICATE_SAME_ACCESS | DUPLICATE_CLOSE_SOURCE)) ::CloseHandle(local);
    }
    if (s.wake) ::CloseHandle(s.wake);
    s.wake = s.remote_wake = nullptr;
    if (s.thread) ::CloseHandle(s.thread);
    if (s.process) ::CloseHandle(s.process);
    s.process = s.thread = nullptr; s.base = s.pid = s.code = 0; s.owner = s.stopping = false;
    return safe;
}

void ResidentExecutor::enter_critical_section() noexcept {
    state().critical_section.store(true, std::memory_order_release);
}

void ResidentExecutor::leave_critical_section() noexcept {
    state().critical_section.store(false, std::memory_order_release);
}

bool ResidentExecutor::in_critical_section() const noexcept {
    return state().critical_section.load(std::memory_order_acquire);
}
} // namespace pvz::runtime
