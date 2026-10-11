// -----------------------------------------------------------------------------
// 自动化辅助模块 (Automation Helper) - 戴夫对白自动推进
// 注意：本模块是测试与训练的便利工具，不属于 RL 动作空间，不参与任何策略决策与奖励计算。
// Automation helpers are training/testing conveniences and are not part of the RL agent action space.
// -----------------------------------------------------------------------------
#pragma once

#include <cstdint>

namespace pvz::automation {

class DialogueHelper {
public:
    // Resident/Boss intros use checked calls after the per-line cooldown.
    // Boss dialogue must never leave queued Enter keys after CancelIntro.
    static bool advance_ready();
    // 判断当前是否显示 Dave（包括战斗中的砸罐子对话）。
    static bool is_in_dialogue();

    // 调用者持有 ProcessManager::mutex()。仅在已验证的游戏版本安装事件钩子。
    static bool sync_event_hook_nolock(bool enabled);
    static void remove_event_hook_nolock();


private:

    // 原版 CutScene::IsShowingCrazyDave() 的等价只读判断。
    static bool is_dave_talking(uintptr_t board);

    // 预留给后续独立的模态 Dialog automation；Crazy Dave 判断不依赖它。
    static bool has_active_dialog();
};

}  // namespace pvz::automation
