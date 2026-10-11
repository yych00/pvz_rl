
#include "../inc/code.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>

namespace {
struct PendingRemoteCall {
    HANDLE process = nullptr;
    HANDLE thread = nullptr;
    void* code = nullptr;
    DWORD pid = 0;
    std::vector<void*> outputs;
    ~PendingRemoteCall() {
        const DWORD saved_error = ::GetLastError();
        // Never free instructions/output while the remote thread is running.
        const bool stopped = !thread || ::WaitForSingleObject(thread, 0) == WAIT_OBJECT_0;
        if (stopped && process) {
            if (code) ::VirtualFreeEx(process, code, 0, MEM_RELEASE);
            for (void* output : outputs) ::VirtualFreeEx(process, output, 0, MEM_RELEASE);
        }
        if (thread) ::CloseHandle(thread);
        if (process) ::CloseHandle(process);
        ::SetLastError(saved_error);
    }
};
struct RemoteCalls {
    std::mutex mutex;
    std::vector<std::unique_ptr<PendingRemoteCall>> pending;
};
RemoteCalls& remote_calls() { static RemoteCalls calls; return calls; }
void reap_nolock(RemoteCalls& calls) {
    auto& pending = calls.pending;
    pending.erase(std::remove_if(pending.begin(), pending.end(), [](const auto& call) {
        return ::WaitForSingleObject(call->thread, 0) == WAIT_OBJECT_0 ||
               ::WaitForSingleObject(call->process, 0) == WAIT_OBJECT_0;
    }), pending.end());
}
bool pending_nolock(const RemoteCalls& calls, DWORD pid) {
    return std::any_of(calls.pending.begin(), calls.pending.end(), [pid](const auto& call) {
        return pid && call->pid == pid;
    });
}
}  // namespace

namespace Pt
{

Code::Code()
{
    unsigned int page = 256; // 1MB
    code = new unsigned char[4096 * page];
    length = 0;
    calls_pos.clear();
}

Code::~Code()
{
    delete[] code;
}

void Code::asm_init()
{
    length = 0;
    calls_pos.clear();
}

void Code::asm_add_byte(unsigned char value)
{
    code[length] = value;
    length += 1;
}

void Code::asm_add_word(unsigned short value)
{
    (unsigned short &)code[length] = value;
    length += 2;
}

void Code::asm_add_dword(unsigned int value)
{
    (unsigned int &)code[length] = value;
    length += 4;
}

void Code::asm_add_list(std::initializer_list<unsigned char> value)
{
    for (auto it = value.begin(); it != value.end(); it++)
        asm_add_byte(*it);
}

void Code::asm_push_byte(unsigned char value)
{
    asm_add_byte(0x6a);
    asm_add_byte(value);
}

void Code::asm_push_dword(unsigned int value)
{
    asm_add_byte(0x68);
    asm_add_dword(value);
}

void Code::asm_mov_exx(Reg reg, unsigned int value)
{
    asm_add_byte(0xb8 + static_cast<unsigned int>(reg));
    asm_add_dword(value);
}

void Code::asm_mov_exx_dword_ptr(Reg reg, unsigned int value)
{
    asm_add_byte(0x8b);
    asm_add_byte(0x05 + static_cast<unsigned int>(reg) * 8);
    asm_add_dword(value);
}

void Code::asm_mov_exx_dword_ptr_exx_add(Reg reg, unsigned int value)
{
    asm_add_byte(0x8b);
    asm_add_byte(0x80 + static_cast<unsigned int>(reg) * (8 + 1));
    if (reg == Reg::ESP)
        asm_add_byte(0x24);
    asm_add_dword(value);
}

void Code::asm_push_exx(Reg reg)
{
    asm_add_byte(0x50 + static_cast<unsigned int>(reg));
}

void Code::asm_pop_exx(Reg reg)
{
    asm_add_byte(0x58 + static_cast<unsigned int>(reg));
}

void Code::asm_mov_exx_exx(Reg reg_to, Reg reg_from)
{
    asm_add_byte(0x8b);
    asm_add_byte(0xc0 + static_cast<unsigned int>(reg_to) * 8 + static_cast<unsigned int>(reg_from));
}

void Code::asm_call(unsigned int addr)
{
    asm_add_byte(0xe8);
    calls_pos.push_back(length);
    asm_add_dword(addr);
}

void Code::asm_ret()
{
    asm_add_byte(0xc3);
}

void Code::asm_code_inject(HANDLE handle)
{
    (void)asm_code_inject(handle, INFINITE);
}

bool Code::asm_code_inject(HANDLE handle, DWORD timeout_ms, DWORD* exit_code)
{
    return execute_remote_code(handle, code, length, timeout_ms, exit_code, calls_pos);
}

bool Code::execute_remote_code(HANDLE process, const unsigned char* bytes, std::size_t size,
                              DWORD timeout_ms, DWORD* exit_code,
                              const std::vector<unsigned int>& calls)
{
    if (!bytes || !size) { ::SetLastError(ERROR_INVALID_PARAMETER); return false; }
    auto& registry = remote_calls();
    std::lock_guard<std::mutex> lock(registry.mutex);
    reap_nolock(registry);
    const DWORD pid = ::GetProcessId(process);
    if (!pid) return false;
    if (pending_nolock(registry, pid)) { ::SetLastError(ERROR_BUSY); return false; }
    auto call = std::make_unique<PendingRemoteCall>();
    call->pid = pid;
    if (!::DuplicateHandle(::GetCurrentProcess(), process, ::GetCurrentProcess(),
                           &call->process, 0, FALSE, DUPLICATE_SAME_ACCESS)) return false;
    call->code = ::VirtualAllocEx(process, nullptr, size,
                                 MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
    if (!call->code) return false;
    std::vector<unsigned char> relocated(bytes, bytes + size);
    for (unsigned int pos : calls) {
        if (pos > size || size - pos < sizeof(int)) {
            ::SetLastError(ERROR_INVALID_PARAMETER); return false;
        }
        unsigned int target = 0;
        std::memcpy(&target, relocated.data() + pos, sizeof(target));
        target -= static_cast<unsigned int>(reinterpret_cast<uintptr_t>(call->code)) + pos + 4;
        std::memcpy(relocated.data() + pos, &target, sizeof(target));
    }
    SIZE_T written = 0;
    if (!::WriteProcessMemory(process, call->code, relocated.data(), size, &written) ||
        written != size || !::FlushInstructionCache(process, call->code, size)) return false;
    // Allocate registry storage BEFORE launching: a bad_alloc cannot strand a live thread.
    registry.pending.push_back(std::move(call));
    auto& active = *registry.pending.back();
    active.thread = ::CreateRemoteThread(process, nullptr, 0,
        reinterpret_cast<LPTHREAD_START_ROUTINE>(active.code), nullptr, 0, nullptr);
    if (!active.thread) {
        const DWORD error = ::GetLastError();
        registry.pending.pop_back(); ::SetLastError(error); return false;
    }
    const DWORD wait = ::WaitForSingleObject(active.thread, timeout_ms);
    if (wait == WAIT_OBJECT_0) {
        DWORD result = 0;
        const bool complete = ::GetExitCodeThread(active.thread, &result) != FALSE;
        const DWORD error = complete ? ERROR_SUCCESS : ::GetLastError();
        if (complete && exit_code) *exit_code = result;
        registry.pending.pop_back(); ::SetLastError(error); return complete;
    }
    // Keep duplicate process/thread handles and code for non-blocking reaping.
    ::SetLastError(wait == WAIT_TIMEOUT ? ERROR_TIMEOUT : ::GetLastError());
    return false;
}

bool Code::has_pending_remote_call(HANDLE process) {
    auto& registry = remote_calls();
    std::lock_guard<std::mutex> lock(registry.mutex);
    reap_nolock(registry);
    return pending_nolock(registry, ::GetProcessId(process));
}

std::size_t Code::reap_remote_calls() {
    auto& registry = remote_calls();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto before = registry.pending.size();
    reap_nolock(registry);
    return before - registry.pending.size();
}

void Code::release_remote_allocation(HANDLE process, void* allocation) noexcept {
    if (!allocation) return;
    try {
        auto& registry = remote_calls();
        std::lock_guard<std::mutex> lock(registry.mutex);
        reap_nolock(registry);
        const DWORD pid = ::GetProcessId(process);
        for (auto& call : registry.pending) {
            if (pid && call->pid == pid) { call->outputs.push_back(allocation); return; }
        }
        ::VirtualFreeEx(process, allocation, 0, MEM_RELEASE);
    } catch (...) {
        // If bookkeeping allocation fails, retaining memory is safer than freeing
        // a live remote thread's output. Process exit will reclaim it.
    }
}

} // namespace Pt
