#pragma once
#include "runtime/pvz_runtime_layout.h"
#include <cstdint>
#include <string>
#include <vector>

namespace pvz::runtime {
// Every caller holds ProcessManager::mutex(). All reads happen while held;
// service() yields only at a known lifecycle boundary.
class ResidentExecutor {
public:
    static ResidentExecutor& instance();
    bool begin(PvzRuntime::VersionedPvZ& pvz, bool owner = true);
    bool stop(PvzRuntime::VersionedPvZ& pvz) noexcept;
    bool owns(const PvzRuntime::VersionedPvZ& pvz) const noexcept;
    bool owner() const noexcept;
    bool execute(const std::uint8_t* bytes, std::size_t size,
                 const std::vector<unsigned int>& calls, std::uint32_t& result);
    bool hold();
    bool service(int milliseconds = 1);
    std::uint64_t commands() const noexcept;
    // Read-only snapshot; never waits for RPC acknowledgement or changes ownership.
    std::string diagnostics_json() const;
    void enter_critical_section() noexcept;
    void leave_critical_section() noexcept;
    bool in_critical_section() const noexcept;
private:
    ResidentExecutor() = default;
};

class CriticalSectionGuard {
public:
    CriticalSectionGuard() noexcept { ResidentExecutor::instance().enter_critical_section(); }
    ~CriticalSectionGuard() noexcept { ResidentExecutor::instance().leave_critical_section(); }
    CriticalSectionGuard(const CriticalSectionGuard&) = delete;
    CriticalSectionGuard& operator=(const CriticalSectionGuard&) = delete;
};
} // namespace pvz::runtime
