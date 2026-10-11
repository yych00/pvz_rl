#pragma once

#if defined(_WIN64) && defined(__cplusplus)
#include <Windows.h>
#include <cstdint>

// 禁用 64 位模式下指针截断与类型转换警告
#pragma warning(disable: 4311 4312 4302 4267 4244)

// 专门针对 32 位目标进程（WoW64）的内存读取拦截重载：
// 在 64 位进程中，原作者代码对多级基址跳转和 PE 头读取使用了 sizeof(uintptr_t) (8 字节)。
// 目标 32 位 PVZ 进程的内部指针与 PE 头偏移只有 4 字节。
// 拦截 8 字节读取请求，精确读取 4 字节并将高 32 位清零，彻底解决 64 位下地址错位！
inline BOOL ReadProcessMemory(HANDLE hProcess, LPCVOID lpBaseAddress, LPVOID lpBuffer, SIZE_T nSize, unsigned long *lpNumberOfBytesRead) {
    if (nSize == 8) {
        uint32_t val32 = 0;
        SIZE_T bytesRead = 0;
        BOOL ret = ::ReadProcessMemory(hProcess, lpBaseAddress, &val32, 4, &bytesRead);
        if (ret && bytesRead == 4) {
            *(uint64_t *)lpBuffer = (uint64_t)val32;
            if (lpNumberOfBytesRead) *lpNumberOfBytesRead = 8;
            return TRUE;
        }
        return FALSE;
    }

    SIZE_T bytes = 0;
    BOOL ret = ::ReadProcessMemory(hProcess, lpBaseAddress, lpBuffer, nSize, &bytes);
    if (lpNumberOfBytesRead) *lpNumberOfBytesRead = (unsigned long)bytes;
    return ret;
}

inline BOOL WriteProcessMemory(HANDLE hProcess, LPVOID lpBaseAddress, LPCVOID lpBuffer, SIZE_T nSize, DWORD *lpNumberOfBytesWritten) {
    SIZE_T bytes = 0;
    BOOL ret = ::WriteProcessMemory(hProcess, lpBaseAddress, lpBuffer, nSize, &bytes);
    if (lpNumberOfBytesWritten) *lpNumberOfBytesWritten = (DWORD)bytes;
    return ret;
}
#endif
