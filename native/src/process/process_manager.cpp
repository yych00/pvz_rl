#include "process_manager.h"
#include "thread_desktop.h"
#include "../automation/automation_controller.h"
#include "../automation/dialogue_helper.h"
#include "../runtime/resident_executor.h"

#include <array>
#include <vector>

namespace pvz::process {

ProcessManager& ProcessManager::instance() {
    static ProcessManager inst;
    return inst;
}

ProcessManager::~ProcessManager() {
    std::lock_guard<std::mutex> lock(mutex_);
    automation::DialogueHelper::remove_event_hook_nolock();
    runtime::ResidentExecutor::instance().stop(pvz_);
}

void ProcessManager::sync_cheat_keys_for_scene_nolock(int scene) {
    if (scene == last_scene_) return;
    if (scene != last_scene_) {
        last_scene_ = scene;
    }
    if (cheat_keys_enabled_) {
        PvzRuntime::set_game_cheat_keys(pvz_, true);
    }
}

void ProcessManager::on_attached_nolock() {
    pvz_.BackgroundRunning(true);
    if (no_crazy_dave_) {
        apply_no_crazy_dave_patches_nolock(true);
    }
    apply_crash_prevention_patches_nolock();
    const auto cfg = automation::AutomationController::get_config();
    (void)automation::DialogueHelper::sync_event_hook_nolock(cfg.enabled && cfg.auto_dialogue);

    if (cheat_keys_enabled_) {
        PvzRuntime::set_game_cheat_keys(pvz_, true);
    }
    last_scene_ = pvz_.GameUI();
}

bool ProcessManager::ensure_connected_nolock() {
    Pt::Code::reap_remote_calls();
    ensure_default_desktop();
    const bool was_valid = pvz_.IsValid();

    const bool is_on = (pvz_.GameOn() || pvz_.FindPvZ()) && pvz_.GameOn();
    if (!is_on) {
        last_attached_pid_ = 0;
        last_scene_ = -1;
        return false;
    }

    const DWORD current_pid = pvz_.process_id();
    if (!was_valid || current_pid != last_attached_pid_ || last_attached_pid_ == 0) {
        last_attached_pid_ = current_pid;
        on_attached_nolock();
    }
    return true;
}

bool ProcessManager::attach() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ensure_connected_nolock()) {
        return false;
    }
    // ensure_connected_nolock applies hooks once per PID, including reconnects.
    return true;
}

bool ProcessManager::is_game_on() {
    std::lock_guard<std::mutex> lock(mutex_);
    return ensure_connected_nolock();
}

int ProcessManager::get_ui() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ensure_connected_nolock()) {
        return -1;
    }
    const int scene = pvz_.GameUI();
    sync_cheat_keys_for_scene_nolock(scene);
    return scene;
}

PvzRuntime::VersionedPvZ& ProcessManager::runtime() {
    return pvz_;
}

std::mutex& ProcessManager::mutex() {
    return mutex_;
}

bool ProcessManager::is_battle_ready() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ensure_connected_nolock()) {
        return false;
    }
    const int scene = pvz_.GameUI();
    sync_cheat_keys_for_scene_nolock(scene);
    if (scene != 3) return false;
    uintptr_t board = pvz_.ReadMemory<uintptr_t>({pvz_.data().lawn, pvz_.data().board});
    return board != 0;
}

void ProcessManager::set_background_running(bool enable) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ensure_connected_nolock()) {
        pvz_.BackgroundRunning(enable);
    }
}

bool ProcessManager::set_always_on_top(bool enable) {
    HWND hwnd = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ensure_connected_nolock()) return false;
        hwnd = pvz_.game_window();
    }

    if (!hwnd || !::IsWindow(hwnd) || ::IsIconic(hwnd)) return false;

    // Headless training intentionally blocks the PvZ window thread for most of
    // the time. Queue the z-order change instead of synchronously waiting for
    // that thread, and never hold ProcessManager::mutex_ across a window call.
    // The existing headless UI-service window will process this request on its
    // next release (normally within about three seconds).
    return ::SetWindowPos(
        hwnd,
        enable ? HWND_TOPMOST : HWND_NOTOPMOST,
        0, 0, 0, 0,
        SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE |
            SWP_NOOWNERZORDER | SWP_ASYNCWINDOWPOS) != FALSE;
}

bool ProcessManager::set_cheat_keys(bool enable) {
    std::lock_guard<std::mutex> lock(mutex_);
    cheat_keys_enabled_ = enable;
    if (!ensure_connected_nolock()) return false;
    return PvzRuntime::set_game_cheat_keys(pvz_, enable);
}

bool ProcessManager::get_cheat_keys() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ensure_connected_nolock()) return cheat_keys_enabled_;
    return PvzRuntime::get_game_cheat_keys(pvz_);
}

bool ProcessManager::send_key(uint32_t vk_code) {
    HWND hwnd = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ensure_connected_nolock()) return false;

        const int scene = pvz_.GameUI();
        sync_cheat_keys_for_scene_nolock(scene);
        if (vk_code == VK_SPACE && scene == 3) return false;

        hwnd = pvz_.game_window();
    }
    if (!hwnd || !::IsWindow(hwnd)) return false;
    if (::IsIconic(hwnd)) {
        ::ShowWindow(hwnd, SW_SHOWNOACTIVATE);
    }
    // Reserved key lParam bit marks native lifecycle input for the protected
    // resident pump; ordinary user input cannot alter an idle training board.
    constexpr LPARAM lifecycle_input = 0x10000000;
    ::PostMessageW(hwnd, WM_KEYDOWN, vk_code, lifecycle_input);
    ::PostMessageW(hwnd, WM_KEYUP, vk_code, lifecycle_input);
    return true;
}

uintptr_t ProcessManager::get_dialog(int dialog_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ensure_connected_nolock()) return 0;
    return PvzRuntime::get_dialog(pvz_, dialog_id);
}

bool ProcessManager::kill_dialog(int dialog_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ensure_connected_nolock()) return false;
    return PvzRuntime::kill_dialog(pvz_, dialog_id);
}

bool ProcessManager::set_no_crazy_dave(bool enable) {
    std::lock_guard<std::mutex> lock(mutex_);
    no_crazy_dave_ = enable;
    if (ensure_connected_nolock()) {
        apply_no_crazy_dave_patches_nolock(enable);
        const auto cfg = automation::AutomationController::get_config();
        (void)automation::DialogueHelper::sync_event_hook_nolock(cfg.enabled && cfg.auto_dialogue);
    }
    return true;
}

void ProcessManager::apply_no_crazy_dave_patches_nolock(bool enable) {
    if (!pvz_.GameOn()) return;

    struct DavePatch {
        uintptr_t addr;
        std::array<uint8_t, 2> enable_bytes;
        std::array<uint8_t, 2> original_bytes;
    };

    // 可选 Dave 屏蔽：关闭对话入口、Dave 时间轴和选卡提示。
    std::vector<DavePatch> patches;
    if (pvz_.version_id() == PVZ_1_0_0_1051_EN) {
        patches = {
            {0x0043acba, {0xeb, 0x16}, {0x7f, 0x16}}, // 1-5 Dave 分支
            {0x0041f69b, {0xeb, 0x6d}, {0x75, 0x6d}}, // 1-5 教学植物
            {0x0041f68f, {0xeb, 0x79}, {0x7f, 0x79}}, // 1-5 重玩分支
            {0x0043af1a, {0xeb, 0x48}, {0x74, 0x48}}, // mCrazyDaveTime
            {0x0043b82c, {0xeb, 0x6d}, {0x75, 0x6d}}, // CrazyDaveEnter
            {0x0043b8a6, {0xeb, 0x10}, {0x74, 0x10}}, // Dave talk index
            {0x0040bfd5, {0xeb, 0x4c}, {0x74, 0x4c}}, // seed chooser Dave hint
        };
    } else if (pvz_.version_id() == PVZ_1_2_0_1065_EN) {
        patches = {
            {0x0043ad2a, {0xeb, 0x16}, {0x7f, 0x16}}, // 1-5 Dave 分支
            {0x0041f69b, {0xeb, 0x6d}, {0x75, 0x6d}}, // 1-5 教学植物
            {0x0041f68f, {0xeb, 0x79}, {0x7f, 0x79}}, // 1-5 重玩分支
            {0x0043af8a, {0xeb, 0x48}, {0x74, 0x48}}, // mCrazyDaveTime
            {0x0043b89c, {0xeb, 0x6d}, {0x75, 0x6d}}, // CrazyDaveEnter
            {0x0043b916, {0xeb, 0x10}, {0x74, 0x10}}, // Dave talk index
            {0x0040bfd5, {0xeb, 0x4c}, {0x74, 0x4c}}, // seed chooser Dave hint
        };
    }

    for (const auto& p : patches) {
        if (enable) {
            pvz_.WriteMemory<uint8_t, 2>(p.enable_bytes, {p.addr});
        } else {
            pvz_.WriteMemory<uint8_t, 2>(p.original_bytes, {p.addr});
        }
    }

    if (enable) {
        uintptr_t board = pvz_.ReadMemory<uintptr_t>({pvz_.data().lawn, pvz_.data().board});
        if (board) {
            uintptr_t cut_scene = pvz_.ReadMemory<uintptr_t>({board + pvz_.data().cut_scene});
            if (cut_scene) {
                pvz_.WriteMemory<int>({0}, {cut_scene + 0x20});
                pvz_.WriteMemory<int>({-1}, {cut_scene + 0x28});
            }
            if (pvz_.GameMode() == 0 && pvz_.ReadMemory<int>({board + pvz_.data().adventure_level}) == 5) {
                uintptr_t challenge = pvz_.ReadMemory<uintptr_t>({board + pvz_.data().challenge});
                if (challenge) {
                    pvz_.WriteMemory<uint8_t>(1, {challenge + 0x64});
                }
            }
        }
    }
}

bool ProcessManager::get_no_crazy_dave() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ensure_connected_nolock()) {
        return no_crazy_dave_;
    }

    uintptr_t check_addr = 0;
    if (pvz_.version_id() == PVZ_1_0_0_1051_EN) {
        check_addr = 0x0043b8a6;
    } else if (pvz_.version_id() == PVZ_1_2_0_1065_EN) {
        check_addr = 0x0043b916;
    }

    if (check_addr != 0) {
        uint8_t byte = pvz_.ReadMemory<uint8_t>({check_addr});
        const bool is_patched = (byte == 0xeb);
        if (no_crazy_dave_ && !is_patched) {
            apply_no_crazy_dave_patches_nolock(true);
            return true;
        }
        return is_patched;
    }

    return no_crazy_dave_;
}

void ProcessManager::set_auto_cancel_intro(bool enable) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto_cancel_intro_ = enable;
}

bool ProcessManager::get_auto_cancel_intro() {
    std::lock_guard<std::mutex> lock(mutex_);
    return auto_cancel_intro_;
}

void ProcessManager::apply_crash_prevention_patches_nolock() {
    if (!pvz_.GameOn()) return;

    if (pvz_.version_id() == PVZ_1_2_0_1065_EN) {
        // 5-10：Boss 尚未创建时，传送带权重逻辑可能解引用空指针。
        // 跳过 Boss 查询分支，保留默认花盆权重。
        const std::array<uint8_t, 2> conveyor_patch = {0xeb, 0x15};
        pvz_.WriteMemory<uint8_t, 2>(conveyor_patch, {0x0042345f});

        // 5-10：Boss 尚未创建时，蹦极绳索绘制可能访问空 Boss。
        // 直接进入通用绘制路径。
        const std::array<uint8_t, 10> bungee_patch = {
            0xe9, 0xbe, 0x00, 0x00, 0x00,
            0x90, 0x90, 0x90, 0x90, 0x90
        };
        pvz_.WriteMemory<uint8_t, 10>(bungee_patch, {0x0052db9e});

        // 消除失焦时自动弹出暂停对话框 (LawnApp::LostFocus -> DoPauseDialog)
        // 避免在游戏后台加载、切窗口或多任务自动化控制时频繁弹出“游戏暂停”菜单
        const std::array<uint8_t, 1> lost_focus_patch = {0xc3};
        pvz_.WriteMemory<uint8_t, 1>(lost_focus_patch, {0x0044f5a0});

        // Keep DirectDraw surface initialization intact. The Turbo AV at
        // 0x00632696 originates in Attachment allocation, not surface clearing;
        // controlled updates must also run EffectSystem::ProcessDeleteQueue.
    }
}

}  // namespace pvz::process
