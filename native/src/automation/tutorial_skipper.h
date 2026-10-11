// -----------------------------------------------------------------------------
// 教学状态自动归一化 (Tutorial Skipper)
//
// 目标：把一次性的新人教学从 RL/自动化环境中剥离，但不把教学逻辑塞进
// Plant/Shovel 等 Agent 动作，也不强行切换 GameUI/关卡。
//
// 原版 TutorialState 0..30 按状态机族分组：
//
//   0       OFF                         无教学
//   1..4    Level 1 Peashooter          1-1 豌豆教学，4 为完成态
//   5..8    Level 2 Sunflower           1-2 向日葵教学，8 为完成态
//   9..12   More Sun                    更多向日葵教学，12 为完成态
//   13..14  Slot Machine                拉杆教学，14 为完成态
//   15..18  Shovel                      铲子教学，18 为完成态
//   19..21  Zombiquarium                水族馆提示链，无统一 COMPLETED：
//                                            19=提示买潜水僵尸
//                                            20=已买/继续玩法
//                                            21=提示点击奖杯
//   22..27  Zen Garden                  禅境花园教学，27 为教学完成态
//   28..30  Whack-a-Zombie              敲罐/锤僵尸教学，30 为完成态
//
// 归一化策略：
//   1..3  -> 4   并复制原版完成教学时启动刷怪的倒计时副作用
//   5..7  -> 8   若尚未开始第一波，复制原版 999 tick 倒计时副作用
//   9..11 -> 12
//   13    -> 14
//   15..17-> 18  并复制原版 CutScene 的铲子教学收尾字段
//   19    -> 20  只跳过“购买潜水僵尸”的教学提示；不凭空生成僵尸
//   20/21 保留    21 属于真实胜利/奖杯阶段，不自动替 Agent 点击奖杯
//   22..26-> 27  只结束教学状态，不强制导航离开禅境花园
//   28..29-> 30
//
// 重要边界：
// - 这是 Automation Helper，不属于 RL action space / reward / observation。
// - 不猜测 Board::SetTutorialState 的远程函数地址；只写已核对的状态字段和
//   必要副作用，避免错误 CALL 导致进程卡死。
// - 只在有效 Board 且 GameUI=2/3 时工作；终局、奖励、菜单状态不修改。
// -----------------------------------------------------------------------------
#pragma once

namespace pvz::automation {

class TutorialSkipper {
public:
    // 自动跳过总开关；独立于旧三参数 automation ABI，默认开启。
    static void set_enabled(bool enable) noexcept;
    static bool enabled() noexcept;

    // 当前 Board::mTutorialState；无有效/受支持的 Board 时返回 -1。
    static int get_state();

    // 把当前教学状态安全推进到本状态机族的非阻塞/完成态。
    // 返回 true 表示本次确实修改了游戏状态。
    static bool skip_current();

    static const char* state_name(int state) noexcept;
};

}  // namespace pvz::automation
