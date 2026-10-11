
#pragma once

#include <Windows.h>

#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <vector>

namespace Pt
{

typedef void *HANDLE;

enum class Reg : unsigned int
{
    EAX = 0,
    EBX = 3,
    ECX = 1,
    EDX = 2,
    ESI = 6,
    EDI = 7,
    EBP = 5,
    ESP = 4,
};

class Code
{
  public:
    Code();
    ~Code();

    Code(const Code &) = delete;
    Code &operator=(const Code &) = delete;

    void asm_init();

    void asm_add_byte(unsigned char);
    void asm_add_word(unsigned short);
    void asm_add_dword(unsigned int);

    void asm_add_list(std::initializer_list<unsigned char>);
    template <typename... Args>
    void asm_add_list(Args...);

    void asm_push_byte(unsigned char);
    void asm_push_dword(unsigned int);

    void asm_mov_exx(Reg, unsigned int);
    void asm_mov_exx_dword_ptr(Reg, unsigned int);
    void asm_mov_exx_dword_ptr_exx_add(Reg, unsigned int);

    void asm_push_exx(Reg);
    void asm_pop_exx(Reg);
    void asm_mov_exx_exx(Reg, Reg);

    void asm_call(unsigned int);

    void asm_ret();

    void asm_code_inject(HANDLE);
    // On timeout executing code remains mapped; the caller can release any
    // game-thread barrier without freeing instructions still in use.
    bool asm_code_inject(HANDLE, DWORD timeout_ms, DWORD* exit_code = nullptr);

    // Shared by Gym's generated Board::Update stubs. A timed-out thread owns
    // its code and output allocations until a later non-blocking reap.
    static bool execute_remote_code(HANDLE, const unsigned char*, std::size_t,
                                    DWORD timeout_ms, DWORD* exit_code = nullptr,
                                    const std::vector<unsigned int>& calls = {});
    static bool has_pending_remote_call(HANDLE);
    static std::size_t reap_remote_calls();
    static void release_remote_allocation(HANDLE, void*) noexcept;

  protected:
    unsigned char *code;
    unsigned int length;
    std::vector<unsigned int> calls_pos;
};

template <typename... Args>
void Code::asm_add_list(Args... value)
{
    asm_add_list({static_cast<unsigned char>(value)...});
}

} // namespace Pt
