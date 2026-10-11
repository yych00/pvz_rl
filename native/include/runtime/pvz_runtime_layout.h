#pragma once

#include "compat_x64.h"
#include "pvz.h"
#include "checked_memory.h"
#include "remote_allocation.h"
#include <Windows.h>

#include <cstddef>
#include <cstdint>

namespace PvzRuntime {

constexpr uintptr_t kPlantStateOffset = 0x3c;
constexpr uintptr_t kPlantStateCountdownOffset = 0x54;
constexpr int kCobCannonReadyState = 37;

// Native runtime adapter: checked reads and bounded virtual injection, plus version/layout access.
class VersionedPvZ final : public Pt::PvZ {
  public:
    // Caller holds ProcessManager mutex. Borrow only a verified persistent barrier.
    void asm_code_inject() override;
    void asm_code_inject(bool main_loop_already_frozen);
    void ensure_remote_idle();

    int version_id() noexcept
    {
        return find_result;
    }

    HWND game_window() const noexcept
    {
        return hwnd;
    }

    HANDLE process_handle() const noexcept
    {
        return handle;
    }

    DWORD process_id() const noexcept
    {
        return pid;
    }

    // 连续读取远程内存块，避免数百次单标量读取造成跨进程通信瓶颈。
    bool read_remote_bytes(uintptr_t address, void *buffer, std::size_t size) noexcept
    {
        if (!address || !buffer || size == 0 || !IsValid())
            return false;

        SIZE_T bytes_read = 0;
        return ::ReadProcessMemory(handle,
                                   reinterpret_cast<LPCVOID>(address),
                                   buffer,
                                   size,
                                   &bytes_read) != FALSE &&
               bytes_read == size;
    }

    template <typename T>
    T read_memory_checked(std::initializer_list<uintptr_t> offsets) {
        return checked_memory_read<T>([this](uintptr_t address, void* buffer, std::size_t size) {
            return read_remote_bytes(address, buffer, size);
        }, offsets);
    }
};

inline bool set_game_window_always_on_top(VersionedPvZ &pvz, bool enable) noexcept
{
    HWND window = pvz.game_window();
    if (!window || !IsWindow(window) || IsIconic(window))
        return false;
    return SetWindowPos(window, enable ? HWND_TOPMOST : HWND_NOTOPMOST,
                        0, 0, 0, 0,
                        SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOOWNERZORDER) != FALSE;
}

inline bool set_game_cheat_keys(VersionedPvZ &pvz, bool enable) noexcept
{
    if (!pvz.GameOn() && !pvz.FindPvZ())
        return false;
    const uintptr_t lawn = pvz.ReadMemory<uintptr_t>({pvz.data().lawn});
    if (!lawn)
        return false;
    // LawnApp + 0x7F5：原版作弊键开关 mTodCheatKeys。
    pvz.WriteMemory<uint8_t>(static_cast<uint8_t>(enable ? 1 : 0), {pvz.data().lawn, 0x7F5});
    return true;
}

inline bool get_game_cheat_keys(VersionedPvZ &pvz) noexcept
{
    if (!pvz.GameOn() && !pvz.FindPvZ())
        return false;
    const uintptr_t lawn = pvz.ReadMemory<uintptr_t>({pvz.data().lawn});
    if (!lawn)
        return false;
    return pvz.ReadMemory<uint8_t>({pvz.data().lawn, 0x7F5}) != 0;
}

// 场景作弊按键同步：作弊模式全局保持开启。
inline bool sync_game_cheat_keys_for_scene(VersionedPvZ &pvz) noexcept
{
    if (!pvz.GameOn() && !pvz.FindPvZ())
        return false;
    return set_game_cheat_keys(pvz, true);
}

inline std::size_t zombie_struct_size(VersionedPvZ &pvz) noexcept
{
    if (pvz.isGOTY())
        return 0x168;
#ifdef _PVZ_BETA_LEAK_SUPPORT
    if (pvz.version_id() == PVZ_BETA_0_1_1_1014_EN)
        return 0x160;
#endif
    return 0x15c;
}

inline std::size_t grid_item_struct_size(VersionedPvZ &pvz) noexcept
{
#ifdef _PVZ_BETA_LEAK_SUPPORT
    if (pvz.version_id() == PVZ_BETA_0_1_1_1014_EN)
        return 0x8c;
#endif
    return 0xec;
}

inline uintptr_t coin_array_offset(VersionedPvZ &pvz) noexcept
{
    return pvz.isGOTY() ? 0xfc : 0xe4;
}

constexpr std::size_t kCoinStructSize = 0xd8;
constexpr uintptr_t kCoinDeadOffset = 0x38;
constexpr uintptr_t kCoinBeingCollectedOffset = 0x50;
constexpr uintptr_t kCoinTypeOffset = 0x58;
constexpr uintptr_t kCoinPlantTypeOffset = 0x68;

// Coin::Collect 原生入口，调用约定为 __thiscall，Coin* 通过 ECX 传递。
// 1.0.0.1051 的公开反编译地址为 0x00432060。
// 1.2.0.1065 的 Coin 代码区域整体相对前者偏移 +0x60，因此对应 0x004320C0。
// 未知版本返回 0，禁止通过猜测地址执行远程调用。
inline uintptr_t coin_collect_address(VersionedPvZ &pvz) noexcept
{
    switch (pvz.version_id()) {
    case PVZ_1_0_0_1051_EN:
        return 0x00432060;
    case PVZ_1_2_0_1065_EN:
        return 0x004320c0;
    default:
        return 0;
    }
}

// Board::CanPlantAt 原生种植空间合法性入口。
// 1.0.0.1051 的 0x0040E020 已由公开反编译与 AvZ 调用链交叉验证。
// 1.2.0.1065 的 0x0040E030 由相邻 Board 函数整体 +0x10 映射得到；
// 当前仅作为影子验证器使用，旧规则仍为权威结果，持续比较一致性后再切换。
// 未知版本绝不回退到其他版本地址，避免错误远程调用破坏游戏。
inline uintptr_t can_plant_at_address(VersionedPvZ &pvz) noexcept
{
    switch (pvz.version_id()) {
    case PVZ_1_0_0_1051_EN:
        return 0x0040e020;
    case PVZ_1_2_0_1065_EN:
        return 0x0040e030;
    default:
        return 0;
    }
}

inline uintptr_t cob_cannon_fire_address(VersionedPvZ &pvz) noexcept
{
    switch (pvz.version_id()) {
    case PVZ_1_2_0_1065_EN:
        return 0x00466ed0;
    default:
        return 0;
    }
}

// LawnApp::PreNewGame 原生入口地址，可从现有界面直接进入指定关卡。
inline uintptr_t pre_new_game_address(VersionedPvZ &pvz) noexcept
{
    switch (pvz.version_id()) {
    case PVZ_1_2_0_1065_EN:
        return 0x0044f6a0;
    case PVZ_1_0_0_1051_EN:
        return 0x0044fe30;
    default:
        return 0;
    }
}

// Verified against the 1.2.0.1065 executable: app argument on stack, ret 4.
// Do not guess an entry point for other builds.
inline uintptr_t process_safe_delete_list_address(VersionedPvZ& pvz) noexcept
{
    return pvz.version_id() == PVZ_1_2_0_1065_EN ? 0x00551b20 : 0;
}

// LawnApp::SeedTypeAvailable 原生入口，精确返回玩家是否已解锁或允许使用指定植物。
// 调用约定：EDI=LawnApp*，EBX=SeedType，返回 AL（1 或 0）。
inline uintptr_t seed_type_available_address(VersionedPvZ &pvz) noexcept
{
    switch (pvz.version_id()) {
    case PVZ_1_2_0_1065_EN:
        return 0x00453c60;
    case PVZ_1_0_0_1051_EN:
        return 0x00454370;
    default:
        return 0;
    }
}

inline uintptr_t board_update_address(VersionedPvZ &pvz) noexcept
{
    // Board::Update，适用于 PVZ 1.2.0.1065 EN。
    // 调用约定：__thiscall void Board::Update()。
    //
    // IDA 复核方法：
    // 1. 搜索 mMainCounter 附近引用；
    // 2. 找到函数序言后包含 mMainCounter 自增的函数；
    // 3. 确认调用链为 SexyApp::UpdateFrames → WidgetManager::UpdateFrame → Board::Update。
    switch (pvz.version_id()) {
    case PVZ_1_2_0_1065_EN:
        return 0x00415D50;
    default:
        return 0;
    }
}

// Verified in the supplied 1.2.0.1065 executable (LawnApp::UpdateFrames,
// 0x452830 and 0x45285b). The parent frame normally drains Board's dead objects
// before WidgetManager advances Board, then drains EffectSystem afterward.
// Calling Board::Update alone omits both queues and eventually exhausts the
// fixed-capacity Attachment pool. Board queue uses ESI, plain ret; effect queue
// takes EffectSystem* on the stack and returns with ret 4.
inline uintptr_t board_delete_queue_address(VersionedPvZ &pvz) noexcept
{
    return pvz.version_id() == PVZ_1_2_0_1065_EN ? 0x0041bb00 : 0;
}

inline uintptr_t effect_delete_queue_address(VersionedPvZ &pvz) noexcept
{
    return pvz.version_id() == PVZ_1_2_0_1065_EN ? 0x00445690 : 0;
}

// LawnApp::GetDialog 原生入口，对应 SexyAppBase::GetDialog 虚表项 73。
// 调用约定：__thiscall，ECX=LawnApp*，压入 theDialogId，返回时清理 4 字节参数。
inline uintptr_t get_dialog_address(VersionedPvZ &pvz) noexcept
{
    switch (pvz.version_id()) {
    case PVZ_1_2_0_1065_EN:
        return 0x00546320;
    default:
        return 0;
    }
}

// LawnApp::KillDialog 原生入口，对应 SexyAppBase::KillDialog 虚表项 77。
// 调用约定：__thiscall，ECX=LawnApp*，压入 theDialogId，返回时清理 4 字节参数。
inline uintptr_t kill_dialog_address(VersionedPvZ &pvz) noexcept
{
    switch (pvz.version_id()) {
    case PVZ_1_2_0_1065_EN:
        return 0x00451940;
    default:
        return 0;
    }
}

// 查询指定 ID 的 Dialog* 是否存在；返回远程 Dialog* 地址，未找到或失败时返回 0。
inline uintptr_t get_dialog(VersionedPvZ &pvz, int dialog_id)
{
    if (!pvz.GameOn() && !pvz.FindPvZ())
        return 0;
    const uintptr_t fn = get_dialog_address(pvz);
    if (!fn)
        return 0;
    const uintptr_t lawn = pvz.ReadMemory<uintptr_t>({pvz.data().lawn});
    if (!lawn)
        return 0;

    HANDLE hProc = pvz.process_handle();
    if (!hProc)
        return 0;

    void *remote_out = VirtualAllocEx(hProc, nullptr, 4, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!remote_out)
        throw NativeOperationError(pvz::ActionError::Unsupported, "dialog result allocation failed");
    RemoteAllocation allocation(hProc, remote_out);

    pvz.asm_init();
    pvz.asm_push_dword(static_cast<unsigned int>(dialog_id));
    pvz.asm_mov_exx_dword_ptr(Pt::Reg::ECX, pvz.data().lawn);
    pvz.asm_call(fn);
    // 将 eax 写入 remote_out 指向的远程内存。
    pvz.asm_add_byte(0xA3);
    pvz.asm_add_dword(static_cast<unsigned int>(reinterpret_cast<uintptr_t>(remote_out)));
    pvz.asm_ret();
    pvz.asm_code_inject();

    uint32_t dialog_ptr = 0;
    if (!pvz.read_remote_bytes(reinterpret_cast<uintptr_t>(remote_out), &dialog_ptr, 4))
        throw NativeOperationError(pvz::ActionError::Unsupported, "dialog result read failed");

    return dialog_ptr;
}

// 销毁或关闭指定 ID 的 Dialog。
inline bool kill_dialog(VersionedPvZ &pvz, int dialog_id)
{
    if (!pvz.GameOn() && !pvz.FindPvZ())
        return false;
    const uintptr_t fn = kill_dialog_address(pvz);
    if (!fn)
        return false;
    const uintptr_t lawn = pvz.ReadMemory<uintptr_t>({pvz.data().lawn});
    if (!lawn)
        return false;

    pvz.asm_init();
    pvz.asm_push_dword(static_cast<unsigned int>(dialog_id));
    pvz.asm_mov_exx_dword_ptr(Pt::Reg::ECX, pvz.data().lawn);
    pvz.asm_call(fn);
    // KillDialog defines a bool in AL only. Its high EAX bits can retain -1
    // during a battle reset; the injected thread must return a clean status.
    pvz.asm_add_list(0x31, 0xc0);
    pvz.asm_ret();
    pvz.asm_code_inject();
    return true;
}

// Board::HasConveyorBelt 原生传送带关卡判断入口。
// 调用约定：EAX = Board*，返回 AL (1 为传送带，0 为普通)。
inline uintptr_t has_conveyor_belt_address(VersionedPvZ &pvz) noexcept
{
    switch (pvz.version_id()) {
    case PVZ_1_0_0_1051_EN:
        return 0x0041be70;
    case PVZ_1_2_0_1065_EN:
        return 0x0041be80;
    default:
        return 0;
    }
}

// 通过注入执行原生 Board::HasConveyorBelt 查询当前是否为传送带关卡。
inline bool query_native_has_conveyor_belt_unlocked(VersionedPvZ &pvz, uintptr_t board)
{
    if (!board || !pvz.GameOn() || !pvz.process_handle()) return false;
    const uintptr_t fn = has_conveyor_belt_address(pvz);
    if (!fn) return false;

    HANDLE hProc = pvz.process_handle();
    void *remote_out = VirtualAllocEx(hProc, nullptr, 4, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!remote_out) throw NativeOperationError(pvz::ActionError::Unsupported, "conveyor result allocation failed");
    RemoteAllocation allocation(hProc, remote_out);

    pvz.asm_init();
    pvz.asm_mov_exx(Pt::Reg::EAX, static_cast<unsigned int>(board));
    pvz.asm_call(fn);
    // mov [remote_out], al
    pvz.asm_add_byte(0xA2);
    pvz.asm_add_dword(static_cast<unsigned int>(reinterpret_cast<uintptr_t>(remote_out)));
    pvz.asm_ret();
    pvz.asm_code_inject();

    uint8_t result = 0;
    if (!pvz.read_remote_bytes(reinterpret_cast<uintptr_t>(remote_out), &result, 1))
        throw NativeOperationError(pvz::ActionError::Unsupported, "conveyor result read failed");

    return result != 0;
}

} // 命名空间 PvzRuntime
