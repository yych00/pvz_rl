#pragma once

#include "runtime/pvz_runtime_layout.h"

#include <mutex>
#include <string>

namespace pvz::process {

// 进程管理单例与句柄状态。
class ProcessManager {
public:
    static ProcessManager& instance();

    // 附加游戏进程。
    bool attach();

    // 检查游戏是否在线且可用。
    bool is_game_on();

    // 返回 LawnApp::mGameScene 原始值。
    // 场景 2 同时覆盖开场、对话和选卡，阶段判断应使用 RuntimePhaseReader。
    int get_ui();

    // 获取底层游戏对象引用。
    PvzRuntime::VersionedPvZ& runtime();

    // 获取底层互斥锁。
    std::mutex& mutex();

    // 检查是否已进入可操作战斗。
    bool is_battle_ready();

    // 设置失焦后继续运行。
    void set_background_running(bool enable);

    // 设置游戏窗口置顶。
    bool set_always_on_top(bool enable);

    // 游戏内置作弊键开关，对应 LawnApp + 0x7F5。
    bool set_cheat_keys(bool enable);
    bool get_cheat_keys();

    // 向游戏主窗口发送按键事件。
    bool send_key(uint32_t vk_code);

    // LawnApp 弹窗查询与销毁。
    uintptr_t get_dialog(int dialog_id);
    bool kill_dialog(int dialog_id);

    // 场景切换时同步作弊键；调用者需持有 mutex_。
    void sync_cheat_keys_for_scene_nolock(int scene);

    // 戴夫屏蔽控制。
    bool set_no_crazy_dave(bool enable);
    bool get_no_crazy_dave();
    bool no_crazy_dave_nolock() const noexcept { return no_crazy_dave_; }

    // 开场快进控制。
    void set_auto_cancel_intro(bool enable);
    bool get_auto_cancel_intro();

    // 应用关卡防崩修复；当前覆盖 5-10 的 Boss 空指针路径。
    void apply_crash_prevention_patches_nolock();

private:
    ProcessManager() = default;
    ~ProcessManager();

    // 保证连接并完成附加初始化；调用者需持有 mutex_。
    bool ensure_connected_nolock();
    void on_attached_nolock();
    void apply_no_crazy_dave_patches_nolock(bool enable);

    std::mutex mutex_;
    PvzRuntime::VersionedPvZ pvz_;
    DWORD last_attached_pid_ = 0;
    int last_scene_ = -1;
    bool no_crazy_dave_ = false;
    bool cheat_keys_enabled_ = true;
    bool auto_cancel_intro_ = true;
};

}  // namespace pvz::process
