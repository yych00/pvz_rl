#include "shared_result.h"
#include <Windows.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
namespace pvz::api {
namespace {
struct Mapping {
    HANDLE handle = nullptr;
    char* view = nullptr;
    std::size_t size = 0;
    std::uint64_t revision = 0;
    std::string name;
    ~Mapping() { if (view) ::UnmapViewOfFile(view); if (handle) ::CloseHandle(handle); }
};
thread_local Mapping mapping;
}
std::string shared_result(const std::string& bytes) {
    auto& m = mapping;
    const auto needed = bytes.size() + 12;
    if (needed > 16 * 1024 * 1024) throw std::runtime_error("runtime shared result exceeds 16 MiB");
    if (needed > m.size) {
        if (m.view) ::UnmapViewOfFile(m.view); if (m.handle) ::CloseHandle(m.handle);
        m.view = nullptr; m.handle = nullptr;
        m.size = (needed + 65535) & ~std::size_t(65535);
        m.name = "Local\\pvz-gym-" + std::to_string(::GetCurrentProcessId()) + "-" +
            std::to_string(::GetCurrentThreadId()) + "-" + std::to_string(++m.revision);
        m.handle = ::CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0,
            static_cast<DWORD>(m.size), m.name.c_str());
        if (m.handle) m.view = static_cast<char*>(::MapViewOfFile(m.handle, FILE_MAP_ALL_ACCESS, 0, 0, m.size));
        if (!m.view) throw std::runtime_error("runtime shared mapping allocation failed");
    }
    const std::uint64_t revision = ++m.revision;
    const auto length = static_cast<std::uint32_t>(bytes.size());
    std::memset(m.view, 0, 8); std::memcpy(m.view + 8, &length, 4);
    std::memcpy(m.view + 12, bytes.data(), bytes.size()); ::MemoryBarrier();
    std::memcpy(m.view, &revision, 8);
    // Backslashes must be escaped in the small descriptor JSON.
    return "PVZM{\"name\":\"Local\\\\" + m.name.substr(6) + "\",\"size\":" +
        std::to_string(m.size) + ",\"length\":" + std::to_string(length) +
        ",\"revision\":" + std::to_string(revision) + "}";
}
} // namespace pvz::api
