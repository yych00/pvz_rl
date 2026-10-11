#include "board_update_batch_injector.h"
#include "code.h"
#include <algorithm>
namespace pvz::game {
int execute_board_update_batch(HANDLE process, const BoardUpdateBatchCode& stub, std::uint32_t max_updates) {
    if (!stub.size || stub.size > stub.bytes.size()) return -1;
    const auto requested = static_cast<std::uint64_t>(max_updates) * 20;
    const DWORD timeout_ms = static_cast<DWORD>(std::min<std::uint64_t>(60000, std::max<std::uint64_t>(5000, requested)));
    DWORD issued = 0;
    if (!Pt::Code::execute_remote_code(process, stub.bytes.data(), stub.size, timeout_ms, &issued) || issued > max_updates) return -1;
    return static_cast<int>(issued);
}
} // namespace pvz::game
