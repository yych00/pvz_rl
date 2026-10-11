// -----------------------------------------------------------------------------
// 开场过场自动快进与跳过 (Intro Skipper)
//
// 两阶段策略：
//   1. 第一次进入 LEVEL_INTRO：只发送 ESC，让原版 CancelIntro() 把有选卡关推进到选卡；
//      无选卡关则由原版直接进入战斗。
//   2. 选卡确认后：发送第二次 ESC，由原版 CancelIntro()/CutScene::Update
//      完成收尾。保留 DaveTime + 4500 的原生小推车初始化事件。
//
// 不直接改写开场时间；所有调用方在第二阶段统一批量推进原生 Board::Update，
// 先完成初始化再发送 ESC。同关重开也重新执行两阶段状态机。
// 两个 Intro 阶段各自限制失败重试频率，并且仅在仍处于 Intro 时各记录一次失败。
// -----------------------------------------------------------------------------
#pragma once

namespace pvz::automation {

class IntroSkipper {
public:
    // 自动跳过总开关，默认开启
    static void set_enabled(bool enable) noexcept;
    static bool enabled() noexcept;

    // 第一阶段：不改时间并发送 VK_ESCAPE；兼容入口可识别选卡后的第二阶段。
    static bool fast_forward_to_battle();

    // 第二阶段：选卡确认后发送 VK_ESCAPE，保留时间触发的原生初始化。
    static bool fast_forward_after_seed_selection();

    // 轮询驱动：按阶段发送 ESC；选卡确认后也可由 confirm_seeds() 立即触发第二阶段。
    // 返回 true 表示已触发快进或处于选卡等待中。
    static bool update();

    // 手动重置内部状态（关卡切换时）
    static void reset() noexcept;
};

}  // namespace pvz::automation
