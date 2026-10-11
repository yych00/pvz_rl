// -----------------------------------------------------------------------------
// PvZ 1.2.0.1065 CutScene reverse-engineering address notes
//
// Verified on the user's 1.2.0.1065 executable:
//   CutScene::KeyDown(KeyCode)              = 0x0043D2F0  [VERIFIED]
//   CutScene::AdvanceCrazyDaveDialog(bool)  = 0x0043C9C0  [VERIFIED]
//   CutScene::CancelIntro()                 = 0x0043B430  [VERIFIED]//
//
// Still needed / not yet verified for 1.2.0.1065:
//   CutScene::StartSeedChooser()           = TODO
//   CutScene::EndSeedChooser()             = TODO
//
// Public PvZ 1.0.0.1051 decompilation reference addresses only:
//   CutScene::StartLevelIntro()            = 0x0043A900
//   CutScene::CancelIntro()                = 0x0043B3C0
//   CutScene::AdvanceCrazyDaveDialog(bool) = 0x0043C950
//   CutScene::KeyDown(KeyCode)             = 0x0043D280
//
// NOTE: 1.0 addresses are reference points only. Do not derive 1.2 addresses by
// applying a fixed offset; verify each 1.2 function independently before use.
// This block is documentation only and intentionally does not change runtime logic.
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// 自动化辅助模块 (Automation Helper) - 戴夫新对白事件触发 Enter
// 注意：本模块是测试与训练的便利工具，不属于 RL 动作空间，不参与任何策略决策与奖励计算。
// Automation helpers are training/testing conveniences and are not part of the RL agent action space.
// -----------------------------------------------------------------------------
#include "dialogue_helper.h"
#include "../runtime/resident_executor.h"
#include "../process/process_manager.h"
#include "runtime/pvz_runtime_layout.h"

#include <Windows.h>
#include <array>
#include <cstring>
#include <vector>

namespace pvz::automation {

namespace {
constexpr int kSceneLevelIntro = 2;
constexpr int kTimePanRightEnd = 3500;
constexpr uintptr_t kTalkMessage120 = 0x004546B0;
constexpr uintptr_t kPostMessageAIat120 = 0x00652330;
constexpr uintptr_t kLawnAppPointer120 = 0x006A9EC0;
constexpr uint32_t kGameModeOffset120 = 0x7F8;
constexpr uint32_t kGameModeUpsell = 71;
constexpr uint32_t kGameModeBossRevenge = 35;
constexpr std::array<uint8_t, 8> kTalkMessagePrologue120 = {
    0x83, 0xEC, 0x0C, 0x53, 0x8B, 0x5C, 0x24, 0x18
};
DWORD s_hook_pid = 0;
uintptr_t s_hook_code = 0;

bool write_code(HANDLE process, uintptr_t address, const void* bytes, size_t size) {
    DWORD old_protect = 0;
    if (!::VirtualProtectEx(process, reinterpret_cast<void*>(address), size,
                            PAGE_EXECUTE_READWRITE, &old_protect)) return false;
    SIZE_T written = 0;
    const bool ok = ::WriteProcessMemory(process, reinterpret_cast<void*>(address),
                                          bytes, size, &written) && written == size;
    ::FlushInstructionCache(process, reinterpret_cast<void*>(address), size);
    DWORD ignored = 0;
    ::VirtualProtectEx(process, reinterpret_cast<void*>(address), size,
                       old_protect, &ignored);
    return ok;
}

void append_u32(std::vector<uint8_t>& code, uint32_t value) {
    for (int i = 0; i < 4; ++i) code.push_back(static_cast<uint8_t>(value >> (i * 8)));
}

// The hook runs inside the 32-bit game when LawnApp::CrazyDaveTalkMessage
// starts a new line. Queue Enter after the current game call returns to its
// message pump. The resulting next line fires this hook again, until Dave
// leaves. No timer or external state polling drives dialogue progression.
std::vector<uint8_t> make_event_code(uintptr_t remote, HWND hwnd) {
    std::vector<uint8_t> code(kTalkMessagePrologue120.begin(),
                              kTalkMessagePrologue120.end());
    code.push_back(0x9C); // pushfd
    code.push_back(0x60); // pushad
    // In GAMEMODE_UPSELL Enter opens a pause dialog instead of advancing Dave.
    // That mode runs its own timed dialogue, so leave it to the game.
    code.push_back(0xA1); append_u32(code, static_cast<uint32_t>(kLawnAppPointer120));
    code.insert(code.end(), {0x85, 0xC0, 0x0F, 0x84}); // test eax,eax; jz skip
    const size_t no_lawn_jump = code.size(); append_u32(code, 0);
    code.insert(code.end(), {0x81, 0xB8}); // cmp dword ptr [eax+0x7F8],71
    append_u32(code, kGameModeOffset120);
    append_u32(code, kGameModeUpsell);
    code.insert(code.end(), {0x0F, 0x84}); // je skip
    const size_t upsell_jump = code.size(); append_u32(code, 0);
    // CancelIntro synchronously advances every Dave line. In the Boss intro,
    // queued Enter messages then reach a scene with no Dave and call
    // CancelIntro again: each call creates another Boss. Boss dialogue uses
    // advance_ready's checked native call instead of queuing keyboard input.
    code.insert(code.end(), {0x81, 0xB8});
    append_u32(code, kGameModeOffset120);
    append_u32(code, kGameModeBossRevenge);
    code.insert(code.end(), {0x0F, 0x84});
    const size_t boss_mode_jump = code.size(); append_u32(code, 0);
    code.insert(code.end(), {0x83, 0xB8});
    append_u32(code, kGameModeOffset120); code.push_back(0); // Adventure
    code.insert(code.end(), {0x75, 0x00});
    const size_t non_adventure_jump = code.size() - 1;
    code.insert(code.end(), {0x8B, 0x80}); append_u32(code, 0x768); // Board
    code.insert(code.end(), {0x85, 0xC0, 0x74, 0x00});
    const size_t no_board_jump = code.size() - 1;
    code.insert(code.end(), {0x83, 0xB8}); append_u32(code, 0x5550);
    code.push_back(50);
    code.insert(code.end(), {0x0F, 0x84});
    const size_t boss_level_jump = code.size(); append_u32(code, 0);
    code[non_adventure_jump] = static_cast<uint8_t>(code.size() - (non_adventure_jump + 1));
    code[no_board_jump] = static_cast<uint8_t>(code.size() - (no_board_jump + 1));
    for (uint32_t message : {uint32_t(WM_KEYDOWN), uint32_t(WM_KEYUP)}) {
        code.push_back(0x68); append_u32(code, 0x10000000); // lifecycle lParam
        code.insert(code.end(), {0x6A, 0x0D}); // VK_RETURN
        code.push_back(0x68); append_u32(code, message);
        code.push_back(0x68); append_u32(code, static_cast<uint32_t>(reinterpret_cast<uintptr_t>(hwnd)));
        code.insert(code.end(), {0xFF, 0x15}); // call dword ptr [PostMessageA IAT]
        append_u32(code, static_cast<uint32_t>(kPostMessageAIat120));
    }
    const size_t skip = code.size();
    for (const size_t jump : {no_lawn_jump, upsell_jump, boss_mode_jump, boss_level_jump}) {
        const uint32_t displacement = static_cast<uint32_t>(skip - (jump + 4));
        std::memcpy(code.data() + jump, &displacement, sizeof(displacement));
    }
    code.push_back(0x61); // popad
    code.push_back(0x9D); // popfd
    code.push_back(0xE9);
    append_u32(code, static_cast<uint32_t>((kTalkMessage120 + 8) - (remote + code.size() + 4)));
    return code;
}

}

bool DialogueHelper::advance_ready() {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() ||
        pvz.GameUI() != 2 || pvz.version_id() != PVZ_1_2_0_1065_EN) return false;
    const auto lawn = pvz.ReadMemory<uintptr_t>({pvz.data().lawn});
    const auto board = pvz.ReadMemory<uintptr_t>({lawn + pvz.data().board});
    const bool boss_intro = pvz.GameMode() == static_cast<int>(kGameModeBossRevenge) ||
        (pvz.GameMode() == 0 && board &&
         pvz.ReadMemory<int>({board + pvz.data().adventure_level}) == 50);
    if (!runtime::ResidentExecutor::instance().owner() && !boss_intro) return false;
    const auto cutscene = board ? pvz.ReadMemory<uintptr_t>({board + pvz.data().cut_scene}) : 0;
    if (!cutscene || pvz.ReadMemory<int>({lawn + 0x850}) == -1 ||
        pvz.ReadMemory<int>({cutscene + 0x38}) > 0) return false;
    // Verified original CutScene::KeyDown(CutScene*, KeyCode), ret 8. It retains
    // tutorial/store/final-boss side effects rather than editing message IDs.
    pvz.asm_init(); pvz.asm_push_dword(VK_RETURN);
    pvz.asm_push_dword(static_cast<unsigned int>(cutscene));
    pvz.asm_call(0x0043d2f0); pvz.asm_add_list(0x31, 0xc0);
    pvz.asm_ret(); pvz.asm_code_inject();
    return true;
}

bool DialogueHelper::is_dave_talking(uintptr_t board) {
    if (!board) return false;

    auto& mgr = process::ProcessManager::instance();
    // 全局屏蔽戴夫开启时，戴夫对白恒为 false（使用无锁 accessor 避免自死锁）
    if (mgr.no_crazy_dave_nolock()) return false;

    auto& pvz = mgr.runtime();

    // LawnApp 的 Dave 状态与消息编号跨越开场、战斗和其他 Dave 场景。
    const uintptr_t lawn = pvz.ReadMemory<uintptr_t>({pvz.data().lawn});
    if (lawn && pvz.ReadMemory<int>({lawn + 0x844}) != 0 &&
        pvz.ReadMemory<int>({lawn + 0x850}) != -1) return true;

    // Original CutScene::IsShowingCrazyDave() equivalent for level intro:
    //   mApp->mGameScene == SCENE_LEVEL_INTRO &&
    //   mCrazyDaveTime > 0 &&
    //   mCutsceneTime < TimePanRightEnd + mCrazyDaveTime
    if (pvz.GameUI() != kSceneLevelIntro) return false;

    const uintptr_t cut_scene = pvz.ReadMemory<uintptr_t>({board + pvz.data().cut_scene});
    if (!cut_scene) return false;

    const int cutscene_time = pvz.ReadMemory<int>({cut_scene + 0x08});
    const int crazy_dave_time = pvz.ReadMemory<int>({cut_scene + 0x20});

    return crazy_dave_time > 0 &&
           cutscene_time < (kTimePanRightEnd + crazy_dave_time);
}

bool DialogueHelper::has_active_dialog() {
    auto& mgr = process::ProcessManager::instance();
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn()) return false;

    const uintptr_t lawn = pvz.ReadMemory<uintptr_t>({pvz.data().lawn});
    if (!lawn) return false;

    // SexyAppBase::mDialogMap / mDialogList layout is not yet version-verified.
    // Crazy Dave automation deliberately does not depend on this heuristic.
    const uintptr_t dialog_list_head = pvz.ReadMemory<uintptr_t>({lawn + 0x31C});
    if (!dialog_list_head) return false;

    const uintptr_t dialog_ptr = pvz.ReadMemory<uintptr_t>({dialog_list_head});
    return dialog_ptr != 0 && dialog_ptr != dialog_list_head;
}

bool DialogueHelper::is_in_dialogue() {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn()) return false;

    const uintptr_t board = pvz.ReadMemory<uintptr_t>({pvz.data().lawn, pvz.data().board});
    if (!board) return false;

    return is_dave_talking(board);
}

bool DialogueHelper::sync_event_hook_nolock(bool enabled) {
    auto& mgr = process::ProcessManager::instance();
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn()) return !enabled;
    const DWORD pid = pvz.process_id();
    if (pid != s_hook_pid) {
        s_hook_pid = pid;
        s_hook_code = 0;
    }
    if (!enabled || mgr.no_crazy_dave_nolock()) {
        remove_event_hook_nolock();
        return true;
    }
    if (s_hook_code) return true;
    if (pvz.version_id() != PVZ_1_2_0_1065_EN) return false;
    HWND hwnd = pvz.game_window();
    if (!hwnd || !::IsWindow(hwnd)) return false;
    HANDLE process = pvz.process_handle();
    std::array<uint8_t, 8> actual{};
    SIZE_T count = 0;
    if (!::ReadProcessMemory(process, reinterpret_cast<void*>(kTalkMessage120),
                             actual.data(), actual.size(), &count) ||
        count != actual.size() || actual != kTalkMessagePrologue120) return false;

    // Validate the game's imported PostMessageA pointer before injecting a
    // stub that calls it. This address and prologue are verified for 1.2.0.1065.
    uint32_t post_message = 0;
    if (!::ReadProcessMemory(process, reinterpret_cast<void*>(kPostMessageAIat120),
                             &post_message, sizeof(post_message), &count) ||
        count != sizeof(post_message) || !post_message) return false;

    const auto remote = reinterpret_cast<uintptr_t>(::VirtualAllocEx(
        process, nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
    if (!remote) return false;
    const auto code = make_event_code(remote, hwnd);
    if (!write_code(process, remote, code.data(), code.size())) {
        ::VirtualFreeEx(process, reinterpret_cast<void*>(remote), 0, MEM_RELEASE);
        return false;
    }
    std::array<uint8_t, 8> patch = {0xE9};
    const uint32_t jump = static_cast<uint32_t>(remote - (kTalkMessage120 + 5));
    std::memcpy(patch.data() + 1, &jump, sizeof(jump));
    patch[5] = patch[6] = patch[7] = 0x90;
    if (!write_code(process, kTalkMessage120, patch.data(), patch.size())) {
        (void)write_code(process, kTalkMessage120,
                         kTalkMessagePrologue120.data(), kTalkMessagePrologue120.size());
        ::VirtualFreeEx(process, reinterpret_cast<void*>(remote), 0, MEM_RELEASE);
        return false;
    }
    s_hook_code = remote;
    // Enabling the option while a line is already on screen must start the
    // chain; subsequent lines are driven solely by CrazyDaveTalkMessage.
    const uintptr_t board = pvz.ReadMemory<uintptr_t>({pvz.data().lawn, pvz.data().board});
    const bool boss_intro = pvz.GameMode() == static_cast<int>(kGameModeBossRevenge) ||
        (pvz.GameMode() == 0 && board &&
         pvz.ReadMemory<int>({board + pvz.data().adventure_level}) == 50);
    if (board && !boss_intro && pvz.GameMode() != static_cast<int>(kGameModeUpsell) &&
        is_dave_talking(board)) {
        ::PostMessageW(hwnd, WM_KEYDOWN, VK_RETURN, 0x10000000);
        ::PostMessageW(hwnd, WM_KEYUP, VK_RETURN, 0x10000000);
    }
    return true;
}

void DialogueHelper::remove_event_hook_nolock() {
    if (!s_hook_code) return;
    auto& pvz = process::ProcessManager::instance().runtime();
    if (pvz.GameOn() && pvz.process_id() == s_hook_pid) {
        // Leave the tiny remote allocation in place: a game thread may still
        // be returning through it when the hook is switched off.
        std::array<uint8_t, 8> expected = {0xE9};
        const uint32_t jump = static_cast<uint32_t>(s_hook_code - (kTalkMessage120 + 5));
        std::memcpy(expected.data() + 1, &jump, sizeof(jump));
        expected[5] = expected[6] = expected[7] = 0x90;
        std::array<uint8_t, 8> actual{};
        SIZE_T count = 0;
        if (::ReadProcessMemory(pvz.process_handle(), reinterpret_cast<void*>(kTalkMessage120),
                                actual.data(), actual.size(), &count) &&
            count == actual.size() && actual == expected) {
            (void)write_code(pvz.process_handle(), kTalkMessage120,
                             kTalkMessagePrologue120.data(), kTalkMessagePrologue120.size());
        }
    }
    s_hook_code = 0;
}

}  // namespace pvz::automation
