#include "auto_whack.h"

#include "../process/process_manager.h"

#include <Windows.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <vector>

namespace pvz::automation {

bool AutoWhack::whack_one() {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() || pvz.GameUI() != 3) return false;

    const HWND window = pvz.game_window();
    const uintptr_t board = pvz.ReadMemory<uintptr_t>({pvz.data().lawn, pvz.data().board});
    if (!window || !IsWindow(window) || !board ||
        pvz.ReadMemory<bool>({board + pvz.data().game_paused})) return false;

    const int mode = pvz.GameMode();
    const bool is_whack = (mode == 30) ||
        (mode == 0 && pvz.ReadMemory<int>({board + pvz.data().adventure_level}) == 15);
    if (!is_whack) return false;

    const int count = pvz.ReadMemory<int>({board + pvz.data().zombie_count_max});
    const uintptr_t zombies = pvz.ReadMemory<uintptr_t>({board + pvz.data().zombie});
    if (!zombies || count <= 0 || count > 1024) return false;

    const size_t stride = PvzRuntime::zombie_struct_size(pvz);
    float target_x = 801.0f;
    float target_y = 0.0f;
    for (int i = 0; i < count; ++i) {
        const uintptr_t zombie = zombies + static_cast<uintptr_t>(i) * stride;
        const uint32_t id = pvz.ReadMemory<uint32_t>({zombie + stride - sizeof(uint32_t)});
        if ((id & 0xffff0000u) == 0 || (id & 0xffffu) != static_cast<unsigned>(i) ||
            pvz.ReadMemory<bool>({zombie + pvz.data().zombie_dead})) continue;

        const int phase = pvz.ReadMemory<int>({zombie + pvz.data().zombie_status});
        if (phase >= 1 && phase <= 3) continue;
        const float x = pvz.ReadMemory<float>({zombie + 0x2c});
        const float y = pvz.ReadMemory<float>({zombie + 0x30});
        if (!std::isfinite(x) || !std::isfinite(y) ||
            x < 80.0f || x > 760.0f || y < 50.0f || y > 510.0f) continue;
        if (x < target_x) {
            target_x = x;
            target_y = y;
        }
    }
    if (target_x > 800.0f) return false;

    RECT client{};
    if (!GetClientRect(window, &client) || client.right <= 0 || client.bottom <= 0)
        return false;
    const int click_x = std::clamp(
        static_cast<int>((target_x + 40.0f) * client.right / 800.0f),
        0, static_cast<int>(client.right) - 1);
    const int click_y = std::clamp(
        static_cast<int>((target_y + 55.0f) * client.bottom / 600.0f),
        0, static_cast<int>(client.bottom) - 1);
    const LPARAM point = MAKELPARAM(click_x, click_y);
    PostMessageW(window, WM_MOUSEMOVE, 0, point);
    PostMessageW(window, WM_LBUTTONDOWN, MK_LBUTTON, point);
    PostMessageW(window, WM_LBUTTONUP, 0, point);
    return true;
}

int AutoWhack::whack_all() {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() || pvz.GameUI() != 3) return 0;

    const HWND window = pvz.game_window();
    const uintptr_t board = pvz.ReadMemory<uintptr_t>({pvz.data().lawn, pvz.data().board});
    if (!window || !IsWindow(window) || !board ||
        pvz.ReadMemory<bool>({board + pvz.data().game_paused})) return 0;

    const int mode = pvz.GameMode();
    const bool is_whack = (mode == 30) ||
        (mode == 0 && pvz.ReadMemory<int>({board + pvz.data().adventure_level}) == 15);
    if (!is_whack) return 0;

    const int count = pvz.ReadMemory<int>({board + pvz.data().zombie_count_max});
    const uintptr_t zombies = pvz.ReadMemory<uintptr_t>({board + pvz.data().zombie});
    if (!zombies || count <= 0 || count > 1024) return 0;

    RECT client{};
    if (!GetClientRect(window, &client) || client.right <= 0 || client.bottom <= 0)
        return 0;

    struct TargetCoord {
        float x;
        float y;
    };
    std::vector<TargetCoord> targets;
    targets.reserve(16);

    const size_t stride = PvzRuntime::zombie_struct_size(pvz);
    for (int i = 0; i < count; ++i) {
        const uintptr_t zombie = zombies + static_cast<uintptr_t>(i) * stride;
        const uint32_t id = pvz.ReadMemory<uint32_t>({zombie + stride - sizeof(uint32_t)});
        if ((id & 0xffff0000u) == 0 || (id & 0xffffu) != static_cast<unsigned>(i) ||
            pvz.ReadMemory<bool>({zombie + pvz.data().zombie_dead})) continue;

        const int phase = pvz.ReadMemory<int>({zombie + pvz.data().zombie_status});
        if (phase >= 1 && phase <= 3) continue;
        const float x = pvz.ReadMemory<float>({zombie + 0x2c});
        const float y = pvz.ReadMemory<float>({zombie + 0x30});
        if (!std::isfinite(x) || !std::isfinite(y) ||
            x < 80.0f || x > 760.0f || y < 50.0f || y > 510.0f) continue;
        targets.push_back({x, y});
    }

    if (targets.empty()) return 0;

    // 优先砸靠左侧、离家最近的僵尸
    std::sort(targets.begin(), targets.end(), [](const TargetCoord& a, const TargetCoord& b) {
        return a.x < b.x;
    });

    int whacked = 0;
    for (const auto& target : targets) {
        const int click_x = std::clamp(
            static_cast<int>((target.x + 40.0f) * client.right / 800.0f),
            0, static_cast<int>(client.right) - 1);
        const int click_y = std::clamp(
            static_cast<int>((target.y + 55.0f) * client.bottom / 600.0f),
            0, static_cast<int>(client.bottom) - 1);
        const LPARAM point = MAKELPARAM(click_x, click_y);
        PostMessageW(window, WM_MOUSEMOVE, 0, point);
        PostMessageW(window, WM_LBUTTONDOWN, MK_LBUTTON, point);
        PostMessageW(window, WM_LBUTTONUP, 0, point);
        ++whacked;
    }
    return whacked;
}

}  // namespace pvz::automation
