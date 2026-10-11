#include "memory_io.h"
#include "process_manager.h"

namespace pvz::process {

bool MemoryIO::read_bytes(uintptr_t remote_addr, void* buffer, std::size_t size) {
    if (!remote_addr || !buffer || size == 0) return false;
    auto& mgr = ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return false;
    return pvz.read_remote_bytes(remote_addr, buffer, size);
}

uintptr_t MemoryIO::resolve_address(const std::vector<uintptr_t>& offsets) {
    if (offsets.empty()) return 0;
    auto& mgr = ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return 0;

    uintptr_t current = offsets[0];
    for (size_t i = 1; i < offsets.size(); ++i) {
        if (!current) return 0;
        current = pvz.ReadMemory<uintptr_t>({current});
        if (!current) return 0;
        current += offsets[i];
    }
    return current;
}

uintptr_t MemoryIO::get_board_address() {
    auto& mgr = ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) return 0;
    return pvz.ReadMemory<uintptr_t>({pvz.data().lawn, pvz.data().board});
}

}  // namespace pvz::process
