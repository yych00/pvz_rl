#pragma once
#include <Windows.h>
namespace pvz::process {
inline bool owns_live_thread(bool acquired, HANDLE thread, DWORD pid) noexcept {
    return acquired && thread && pid && GetProcessIdOfThread(thread) == pid &&
           WaitForSingleObject(thread, 0) == WAIT_TIMEOUT;
}
}
