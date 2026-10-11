#pragma once

#include <cstdint>
#include <string>

namespace PvzRuntime { class VersionedPvZ; }

namespace pvz::game {

// 游戏内部逻辑时钟与游戏刻推进控制。
class SimulationClock {
public:
    enum class HeadlessBeginStatus : int {
        Ok = 0,
        AlreadyActive = 1,
        GameNotFound = 2,
        NotInBattle = 3,
        BoardNull = 4,
        UpdateFnNull = 5,
        BattleChanged = 6,
        BarrierAcquireFailed = 7,
        UiWorkerStartFailed = 8,
    };

    // 暂停战斗逻辑，设置 Board::mPaused。
    // 暂停后 Board::Update 会提前返回，因此不能用这种方式直接做 RL 快进。
    static bool pause();

    // 恢复游戏运行。
    static bool resume();

    // 检查当前战斗是否暂停。
    static bool is_paused();

    // 注入调用一次 Board::Update，驱动游戏逻辑走一帧。
    // 调用前必须处于战斗状态且 Board::mPaused 为 false。
    static bool step_one_tick();

    // 快进指定数量的战斗逻辑游戏刻，以 mMainCounter 的真实变化量为准，不依赖现实时间。
    // 返回实际推进量；失败时返回负数错误码。
    static int advance_ticks(int ticks);

    // 持久无头模式在启动时接管原版窗口主线程，并在经过验证的屏障处保持线程挂起；
    // RL 通过注入 Board::Update 连续推进，决策之间保持 Board 暂停。
    // 独立看门狗约每三秒短暂释放主线程处理界面和消息泵，再重新取得屏障，
    // 因此 Python 计算或暂停期间也不会让窗口长期失去响应。
    // advance_ticks_headless 返回的仍是权威 game_clock 实际增量，教程或终局切换附近不要求严格等于请求值。
    static bool begin_headless_mode();
    static int advance_ticks_headless(int ticks);
    static bool end_headless_mode();

    // 仅当持久无头模式已经挂起原生主线程时为 true。
    // 已经持有 ProcessManager 互斥锁的辅助函数可据此避免重复获取 MainLoopBarrier。
    static bool headless_owns_main_loop() noexcept;
    // Caller must hold ProcessManager mutex; checks live process, battle and barrier ownership.
    static bool can_reuse_headless_barrier_nolock(PvzRuntime::VersionedPvZ& pvz) noexcept;
    static void release_headless_after_failure_nolock(PvzRuntime::VersionedPvZ& pvz) noexcept;

    // 返回最近一次无头模式启动结果和当前所有权/界面服务统计。
    static int headless_begin_status();
    static std::string headless_diagnostics_json();

    // 快进关卡开场阶段的 Board::Update 次数，供 Crazy Dave 和战前自动化使用。
    // CutScene::Update 会随 Board::Update 推进，因此无需按现实毫秒等待。
    // 返回值是实际执行的 Board::Update 次数，不是 mMainCounter 增量。
    static int advance_intro_ticks(int ticks);
    // LIVE has no environment wait loop. Advance the closed chooser through
    // the original initialization event before its second ESC (1065 only).
    // Returns zero outside that window; never advances an open seed chooser.
    static int advance_post_seed_intro();
    // Loss-cutscene updates do not advance the authoritative battle clock.
    static int advance_terminal_updates(int updates);

    // 获取来自 Board::mMainCounter 的权威游戏逻辑时钟。
    static std::int64_t get_game_clock();
};

}  // 命名空间 pvz::game
