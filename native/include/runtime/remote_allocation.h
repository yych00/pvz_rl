#pragma once

#include "code.h"

namespace PvzRuntime {
// Result buffers follow the same lifetime as timed-out remote instructions.
class RemoteAllocation {
public:
    RemoteAllocation(HANDLE process, void* address) noexcept : process_(process), address_(address) {}
    ~RemoteAllocation() { Pt::Code::release_remote_allocation(process_, address_); }
    RemoteAllocation(const RemoteAllocation&) = delete;
    RemoteAllocation& operator=(const RemoteAllocation&) = delete;
private:
    HANDLE process_;
    void* address_;
};
}  // namespace PvzRuntime
