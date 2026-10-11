// -----------------------------------------------------------------------------
// 自动化辅助模块 (Automation Helper) - 总控制器实现
//
// Automation helpers are training/testing conveniences and are not part of the RL agent action space.
// -----------------------------------------------------------------------------
#include "automation_controller.h"
#include "dialogue_helper.h"
#include "auto_collect.h"
#include "auto_whack.h"
#include "tutorial_skipper.h"
#include "intro_skipper.h"
#include "../process/process_manager.h"

#include <mutex>

namespace pvz::automation {

namespace {
std::mutex g_config_mutex;
AutomationConfig g_config = {
    /* enabled */       true,
    /* auto_dialogue */ true,
    /* auto_collect */  true,
    /* auto_whack */    true
};
}

void AutomationController::set_config(const AutomationConfig& config) {
    {
        std::lock_guard<std::mutex> lock(g_config_mutex);
        g_config = config;
    }
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    (void)DialogueHelper::sync_event_hook_nolock(config.enabled && config.auto_dialogue);
}

AutomationConfig AutomationController::get_config() {
    std::lock_guard<std::mutex> lock(g_config_mutex);
    return g_config;
}

bool AutomationController::handle_prebattle() {
    // 原生开场自动快进（独立开关，默认开启）
    if (IntroSkipper::enabled()) {
        IntroSkipper::update();
    }

    const auto cfg = get_config();
    if (cfg.enabled && cfg.auto_dialogue) DialogueHelper::advance_ready();
    return true;
}

int AutomationController::update() {
    AutomationConfig cfg = get_config();
    if (!cfg.enabled) {
        return 0;
    }

    // 原生开场自动快进
    if (IntroSkipper::enabled()) {
        IntroSkipper::update();
    }

    // 教学状态属于环境生命周期，不是 Agent 动作。每次统一 automation
    // update 都先解除其阻塞，避免只在 GUI 守护线程中生效。
    if (TutorialSkipper::enabled()) {
        TutorialSkipper::skip_current();
    }

    const int collected = cfg.auto_collect ? AutoCollect::collect_all() : 0;
    if (cfg.auto_whack) {
        AutoWhack::whack_all();
    }
    return collected;
}

}  // namespace pvz::automation
