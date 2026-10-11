#include "intro_skipper.h"
#include "../process/process_manager.h"
#include "../game/simulation_clock.h"

#include <atomic>
#include <chrono>
#include <iostream>

namespace pvz::automation {
namespace {

std::atomic_bool g_auto_skip_intro{true};

// 跟踪当前关卡是否已经历过选卡，避免无选卡关卡误触第二阶段。
bool s_had_seed_selection = false;
bool s_stage1_triggered   = false;
bool s_stage2_triggered   = false;
bool s_stage1_failure_reported = false;
bool s_stage2_failure_reported = false;
int  s_last_level_id      = -1;
uintptr_t s_last_board = 0;
uintptr_t s_last_cut_scene = 0;
int s_stage2_time = -1;
std::chrono::steady_clock::time_point s_next_stage1_retry{};
std::chrono::steady_clock::time_point s_next_stage2_retry{};
constexpr auto kRetryInterval = std::chrono::milliseconds(250);

bool boss_intro(PvzRuntime::VersionedPvZ& pvz, uintptr_t board) {
    return pvz.GameMode() == 35 || (pvz.GameMode() == 0 &&
        pvz.ReadMemory<int>({board + pvz.data().adventure_level}) == 50);
}

bool is_post_seed_window() {
    auto& mgr = process::ProcessManager::instance();
    if (!mgr.is_game_on()) return false;

    auto& pvz = mgr.runtime();
    if (pvz.GameUI() != 2) return false;

    const uintptr_t lawn = pvz.ReadMemory<uintptr_t>({pvz.data().lawn});
    if (!lawn) return false;
    const uintptr_t board = pvz.ReadMemory<uintptr_t>({lawn + pvz.data().board});
    if (!board) return false;
    if (boss_intro(pvz, board)) return false; // Boss has no seed chooser.
    const uintptr_t cut_scene = pvz.ReadMemory<uintptr_t>({board + pvz.data().cut_scene});
    if (!cut_scene) return false;

    const bool seed_choosing = pvz.ReadMemory<bool>({cut_scene + 0x2C});
    if (seed_choosing) return false;

    const int cutscene_time = pvz.ReadMemory<int>({cut_scene + 0x08});
    const int crazy_dave_time = pvz.ReadMemory<int>({cut_scene + 0x20});

    // EndSeedChooser() 将时间放回 DaveTime + 4260 左右。
    // 用 >= DaveTime + 4250 识别选卡后的时间段；第一次 Intro 正常不会在这里调用。
    return cutscene_time >= crazy_dave_time + 4250;
}

bool send_escape(bool accelerate_post_seed) {
    auto& mgr = process::ProcessManager::instance();
    if (!mgr.is_game_on()) return false;

    auto& pvz = mgr.runtime();
    const int scene = pvz.GameUI();
    if (scene != 2) return false;  // 只在 SCENE_LEVEL_INTRO 有效

    const uintptr_t lawn = pvz.ReadMemory<uintptr_t>({pvz.data().lawn});
    if (!lawn) return false;
    const uintptr_t board = pvz.ReadMemory<uintptr_t>({lawn + pvz.data().board});
    if (!board) return false;
    if (accelerate_post_seed && boss_intro(pvz, board)) return false;
    const uintptr_t cut_scene = pvz.ReadMemory<uintptr_t>({board + pvz.data().cut_scene});
    if (!cut_scene) return false;

    const int level_id = pvz.ReadMemory<int>({board + pvz.data().adventure_level});
    // All callers share the same post-chooser fast-forward: run each original
    // update through initialization, then issue ESC. LIVE needs no env wait loop.
    if (accelerate_post_seed) {
        (void)game::SimulationClock::advance_post_seed_intro();
        if (pvz.GameUI() != 2 ||
            pvz.ReadMemory<uintptr_t>({lawn + pvz.data().board}) != board ||
            pvz.ReadMemory<uintptr_t>({board + pvz.data().cut_scene}) != cut_scene) {
            return false;
        }
    }
    // Do not assign mCutsceneTime here. In 1.2.0.1065, CutScene::Update
    // (0x43bea1) calls PlaceStreetZombies at CrazyDaveTime + 4500 exactly;
    // that function initializes the original lawn mowers. Jumping from the
    // post-chooser time (~4260) to 4510 skips that event and starts a battle
    // with no mowers. CancelIntro and the normal/batched Board updates must
    // execute the original initialization before completing the intro.

    // A queued ESC can itself run CancelIntro before that exact-time event.
    // In the verified 1.2.0.1065 layout, +0x36 is mPlacedZombies, set by
    // PlaceStreetZombies (0x43a709) together with InitLawnMowers. Let the
    // intro update through that event before posting the post-chooser ESC.
    // Other versions follow their natural intro instead of using this offset.
    if (accelerate_post_seed &&
        (pvz.version_id() != PVZ_1_2_0_1065_EN ||
         !pvz.read_memory_checked<bool>({cut_scene + 0x36}))) {
        return false;
    }

    const bool sent = mgr.send_key(VK_ESCAPE);
    bool& failure_reported = accelerate_post_seed
        ? s_stage2_failure_reported : s_stage1_failure_reported;
    if (sent) {
        failure_reported = false;
        s_last_board = board;
        s_last_cut_scene = cut_scene;
        if (accelerate_post_seed)
            s_stage2_time = pvz.ReadMemory<int>({cut_scene + 0x08});
    } else if (mgr.is_game_on() && pvz.GameUI() == 2 && !failure_reported) {
        // 切换界面后的失败是正常竞态；只有仍在 Intro 时才报告一次。
        std::cerr << "[FASTINTRO] ESC send failed"
                  << " stage=" << (accelerate_post_seed ? 2 : 1)
                  << " level=" << level_id << '\n';
        failure_reported = true;
    }
    return sent;
}

}  // namespace

void IntroSkipper::set_enabled(bool enable) noexcept {
    g_auto_skip_intro.store(enable, std::memory_order_relaxed);
}

bool IntroSkipper::enabled() noexcept {
    return g_auto_skip_intro.load(std::memory_order_relaxed);
}

void IntroSkipper::reset() noexcept {
    s_had_seed_selection = false;
    s_stage1_triggered   = false;
    s_stage2_triggered   = false;
    s_stage1_failure_reported = false;
    s_stage2_failure_reported = false;
    s_next_stage1_retry = {};
    s_next_stage2_retry = {};
    s_last_board = 0;
    s_last_cut_scene = 0;
    s_stage2_time = -1;
}

// 兼容现有调用：第一次 Intro 只发 ESC；如果检测到已经处于选卡后的时间段，
// 自动切换为第二阶段；时间由原版 CutScene::Update 推进。
bool IntroSkipper::fast_forward_to_battle() {
    if (!enabled()) return false;
    const bool post_seed = is_post_seed_window();
    const bool sent = send_escape(post_seed);
    if (sent) {
        if (post_seed) {
            s_had_seed_selection = true;
            s_stage2_triggered = true;
        } else {
            s_stage1_triggered = true;
        }
    }
    return sent;
}

// 显式第二阶段入口：统一批量推进原生初始化，再发 ESC。
bool IntroSkipper::fast_forward_after_seed_selection() {
    if (!enabled()) return false;
    s_had_seed_selection = true;
    const bool sent = send_escape(true);
    if (sent) s_stage2_triggered = true;
    return sent;
}

// 轮询驱动：第一次进入 LEVEL_INTRO 时发送第一次 ESC。
// 有选卡关卡会进入 SeedSelection；确认选卡后立即尝试第二次，失败则由这里重试。
bool IntroSkipper::update() {
    if (!enabled()) return false;

    auto& mgr = process::ProcessManager::instance();
    if (!mgr.is_game_on()) {
        reset();
        return false;
    }

    auto& pvz = mgr.runtime();
    const int scene = pvz.GameUI();
    if (scene != 2) {
        reset();
        return false;
    }

    const uintptr_t lawn = pvz.ReadMemory<uintptr_t>({pvz.data().lawn});
    if (!lawn) return false;
    const uintptr_t board = pvz.ReadMemory<uintptr_t>({lawn + pvz.data().board});
    if (!board) return false;

    // 关卡切换时重置状态。
    const int level_id = pvz.ReadMemory<int>({board + pvz.data().adventure_level});
    if (level_id != s_last_level_id) {
        reset();
        s_last_level_id = level_id;
    }

    const uintptr_t cut_scene = pvz.ReadMemory<uintptr_t>({board + pvz.data().cut_scene});
    if (!cut_scene) return false;

    // A same-level replay can replace/reuse Board and CutScene without this
    // helper observing the intervening battle. Level ID alone is insufficient:
    // clear the previous episode's sent flags on replacement or clock rewind.
    const int cutscene_time = pvz.ReadMemory<int>({cut_scene + 0x08});
    if ((s_last_board && (s_last_board != board || s_last_cut_scene != cut_scene)) ||
        (s_stage2_triggered && s_stage2_time >= 0 && cutscene_time < s_stage2_time)) {
        reset();
    }
    s_last_board = board;
    s_last_cut_scene = cut_scene;

    const bool seed_choosing = pvz.ReadMemory<bool>({cut_scene + 0x2C});
    if (seed_choosing) {
        // 选卡界面开着：只记录经历过选卡，等待玩家或 Agent 决策。
        s_had_seed_selection = true;
        return true;
    }

    // LIVE polls only LEVEL_INTRO/DIALOGUE, so it can miss the chooser
    // completely. Recognize EndSeedChooser's time window directly instead
    // of requiring a previous poll with mSeedChoosing=true.
    const auto dave_time = static_cast<std::int64_t>(pvz.ReadMemory<int>({cut_scene + 0x20}));
    // BossTime extends this intro past 4250 without ever choosing seeds.
    // A second ESC would call PlayBossEnter again and create a duplicate Boss.
    if (!boss_intro(pvz, board) &&
        static_cast<std::int64_t>(cutscene_time) >= dave_time + 4250)
        s_had_seed_selection = true;

    // 两次 ESC 都发生在 LEVEL_INTRO；选卡关闭后使用独立的第二阶段状态。
    if (s_had_seed_selection && !s_stage2_triggered) {
        const auto now = std::chrono::steady_clock::now();
        if (now < s_next_stage2_retry) return false;
        if (fast_forward_after_seed_selection()) return true;
        s_next_stage2_retry = now + kRetryInterval;
        return false;
    }

    // 第一次 ESC：有选卡关进入选卡，无选卡关直接进入战斗。
    if (!s_stage1_triggered && !s_had_seed_selection) {
        const auto now = std::chrono::steady_clock::now();
        if (now < s_next_stage1_retry) return false;
        if (fast_forward_to_battle()) return true;
        s_next_stage1_retry = now + kRetryInterval;
    }

    return false;
}

}  // namespace pvz::automation
