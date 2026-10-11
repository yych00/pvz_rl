// -----------------------------------------------------------------------------
// 自动化辅助模块 (Automation Helper) - 总控制器
//
// 定位严格声明：
// - 本模块是训练与测试的便利工具 (Convenience Helpers)
// - 不是 RL 模块，不属于 RL 动作空间 (Not part of RL action space)
// - 不参与 Reward 计算，不进入 Action Mask，不改写 Observation 契约
// - 默认开启，但可通过配置完全关闭
//
// Automation helpers are training/testing conveniences and are not part of the RL agent action space.
// -----------------------------------------------------------------------------
#pragma once

#include <cstdint>

namespace pvz::automation {

struct AutomationConfig {
    bool enabled = true;         // 自动化辅助总开关
    bool auto_dialogue = true;   // 自动推进 Crazy Dave 对白与确认弹窗
    bool auto_collect = true;    // 自动收集常规掉落物 (阳光、货币等)
    bool auto_whack = true;      // 冒险 2-5 / 锤僵尸小游戏自动锤击
};

class AutomationController {
public:
    // 配置管理
    static void set_config(const AutomationConfig& config);
    static AutomationConfig get_config();

    // 关卡开始后推进原版开场过场。Dave 对话由游戏事件钩子处理。
    static bool handle_prebattle();

    // 战斗过程中周期性执行自动收集与自动锤击
    // 注意：
    // - update() 不推进游戏 tick
    // - update() 不做任何 RL 策略动作 (Plant/Shovel/CobFire)
    // - update() 不修改 reward
    // - update() 不生成 RL action
    // 返回本次自动收集数量；教学状态归一化不计入返回值。
    static int update();
};

}  // namespace pvz::automation
