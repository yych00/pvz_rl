#pragma once

#include "runtime/compat_x64.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pvz::process {

// 统一健壮内存读写抽象，包含空指针防护与连续大块内存读取
class MemoryIO {
public:
    // 读取单变量 (支持多级指针寻址)
    template <typename T>
    static T read(const std::vector<uintptr_t>& offsets, T default_value = T{});

    // 写入单变量
    template <typename T>
    static bool write(T value, const std::vector<uintptr_t>& offsets);

    // 批量读取连续内存块
    static bool read_bytes(uintptr_t remote_addr, void* buffer, std::size_t size);

    // 解析多级基址指针并返回目标物理地址 (无效返回 0)
    static uintptr_t resolve_address(const std::vector<uintptr_t>& offsets);

    // 读取 Board 指针 (基址 Lawn -> Board)
    static uintptr_t get_board_address();
};

}  // namespace pvz::process
