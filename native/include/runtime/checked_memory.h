#pragma once

#include "native_errors.h"
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <type_traits>

namespace PvzRuntime {
// The target game is 32-bit even when the DLL is 64-bit. Pointer reads are
// explicit uint32_t; an actual double/int64 remains an eight-byte data read.
template <typename T, typename Read>
T checked_memory_read(Read&& read, std::initializer_list<uintptr_t> offsets) {
    static_assert(std::is_trivially_copyable_v<T>);
    auto fail = [](uintptr_t address) -> void {
        std::ostringstream message;
        message << "required native memory read failed at 0x" << std::hex << address;
        throw NativeOperationError(pvz::ActionError::Unsupported, message.str());
    };
    if (!offsets.size()) fail(0);
    uintptr_t base = 0;
    for (auto it = offsets.begin(); it != offsets.end(); ++it) {
        if (base > std::numeric_limits<uint32_t>::max() || *it > std::numeric_limits<uint32_t>::max() - base) fail(base);
        const uintptr_t address = base + *it;
        const size_t width = it + 1 != offsets.end() || std::is_same_v<T, uintptr_t> ? sizeof(uint32_t) : sizeof(T);
        if (!address || width - 1 > std::numeric_limits<uint32_t>::max() - address) fail(address);
        if (it + 1 != offsets.end()) {
            uint32_t pointer = 0;
            if (!read(address, &pointer, sizeof(pointer)) || !pointer) fail(address);
            base = pointer;
        } else {
            if constexpr (std::is_same_v<T, uintptr_t>) {
                uint32_t value = 0;
                if (!read(address, &value, sizeof(value))) fail(address);
                return static_cast<uintptr_t>(value);
            } else {
                T value{};
                if (!read(address, &value, sizeof(value))) fail(address);
                return value;
            }
        }
    }
    fail(base);
    return T{};
}
}  // namespace PvzRuntime
