#include "events_tracker.h"

#include "runtime/game_offsets.h"

#include <algorithm>
#include <unordered_map>

namespace pvz::env {
namespace {

bool is_self_consuming_plant(int type) noexcept {
    using namespace pvz::runtime;
    switch (type) {
        case CherryBomb:
        case PotatoMine:
        case GraveBuster:
        case IceShroom:
        case DoomShroom:
        case Squash:
        case Tanglekelp:
        case Jalapeno:
        case Blover:
        case CoffeeBean:
            return true;
        default:
            return false;
    }
}

const CardState* find_card(const Observation& obs, int slot) noexcept {
    for (const auto& card : obs.cards) {
        if (card.slot == slot) return &card;
    }
    return nullptr;
}

bool has_new_plant_at(
    const Observation& before,
    const Observation& after,
    int row,
    int col) {
    std::unordered_set<int> old_ids;
    for (const auto& plant : before.plants) old_ids.insert(plant.id);
    for (const auto& plant : after.plants) {
        if (plant.row == row && plant.col == col && old_ids.find(plant.id) == old_ids.end()) {
            return true;
        }
    }
    return false;
}

}  // 匿名命名空间

void EventsTracker::begin_step(const Observation& current_obs) {
    baseline_obs_ = current_obs;
    living_zombie_ids_.clear();
    living_plant_ids_.clear();
    for (const auto& zombie : current_obs.zombies) living_zombie_ids_.insert(zombie.id);
    for (const auto& plant : current_obs.plants) living_plant_ids_.insert(plant.id);
    baseline_sun_ = current_obs.global.sun;
    baseline_wave_ = current_obs.global.current_wave;
    has_baseline_ = true;
}

StepEvents EventsTracker::end_step(
    const Observation& new_obs,
    const Action& action,
    const ActionResult& action_result,
    GameStatus new_status) {
    StepEvents events;
    if (!has_baseline_) return events;

    if (new_obs.quality == "terminal_fallback") {
        events.won = baseline_obs_.global.status != GameStatus::Won && new_status == GameStatus::Won;
        events.lost = baseline_obs_.global.status != GameStatus::Lost && new_status == GameStatus::Lost;
        reset();
        return events;
    }

    std::unordered_set<int> new_zombie_ids;
    for (const auto& zombie : new_obs.zombies) new_zombie_ids.insert(zombie.id);

    // 失败终局流程本身可能移除或回收僵尸，不能把这些消失错误统计成智能体击杀。
    if (new_status != GameStatus::Lost) {
        for (int id : living_zombie_ids_) {
            if (new_zombie_ids.find(id) == new_zombie_ids.end()) ++events.zombies_killed;
        }
    }

    std::unordered_set<int> new_plant_ids;
    for (const auto& plant : new_obs.plants) new_plant_ids.insert(plant.id);

    int planted_type = -1;
    if (action_result.success && action.type == ActionType::Plant) {
        if (const auto* card = find_card(baseline_obs_, action.card_slot)) {
            planted_type = card->plant_type;
            events.sun_spent = std::max(0, card->cost);
        }
    }

    for (const auto& old_plant : baseline_obs_.plants) {
        if (new_plant_ids.find(old_plant.id) != new_plant_ids.end()) continue;

        // 樱桃炸弹、土豆雷等一次性植物因为自身效果结束而消失，不算战略性植物损失。
        if (is_self_consuming_plant(old_plant.type)) continue;

        // 成功铲除是智能体主动移除指定植物，不能计入战斗损失。
        if (action_result.success && action.type == ActionType::Shovel &&
            old_plant.row == action.row && old_plant.col == action.col) {
            continue;
        }

        // 升级植物会用新 ID 替换底座植物；玉米加农炮还会同时消耗右侧第二棵玉米投手。
        // 这些主动替换不应被统计为植物损失。
        if (action_result.success && action.type == ActionType::Plant &&
            old_plant.row == action.row) {
            const bool target_replaced =
                old_plant.col == action.col &&
                has_new_plant_at(baseline_obs_, new_obs, action.row, action.col);
            const bool cob_second_base =
                planted_type == runtime::CobCannon &&
                old_plant.col == action.col + 1 &&
                has_new_plant_at(baseline_obs_, new_obs, action.row, action.col);
            if (target_replaced || cob_second_base) continue;
        }

        ++events.plants_lost;
    }

    // 成功种植的阳光消耗可以从动作精确得到；剩余的正向阳光变化才属于本步真实收入。
    // 这样不会把“收集 50、同时花费 100”错误记录成单纯 -50。
    events.sun_collected = std::max(
        0, new_obs.global.sun - baseline_sun_ + events.sun_spent);

    // READY disappearance is ambiguous: victory sells unused mowers before WON.
    // The persistent game counter also catches activation and removal within
    // one long RL step, without ever counting those cleanup removals.
    if (new_obs.global.board_id == baseline_obs_.global.board_id &&
        new_obs.global.triggered_mowers >= baseline_obs_.global.triggered_mowers) {
        events.mowers_used = new_obs.global.triggered_mowers - baseline_obs_.global.triggered_mowers;
        const auto& before = baseline_obs_.global.mower_trigger_waves;
        const auto& after = new_obs.global.mower_trigger_waves;
        if (events.mowers_used > 0) {
            for (size_t row = 0; row < std::min(before.size(), after.size()); ++row)
                if (after[row] >= 0 && after[row] != before[row])
                    events.mower_triggered_rows.push_back(static_cast<std::int32_t>(row));
        }
        // Boss ice/fire balls call SquishMower without incrementing StartMower's
        // counter. Count a newly squished unused mower once, including a dead
        // slot after a long step; an already triggered mower was paid earlier.
        for (const auto& mower : new_obs.mowers) {
            if (!mower.combat_destroyed || mower.row < 0) continue;
            bool was_squished = false;
            for (const auto& old : baseline_obs_.mowers)
                if (old.row == mower.row) was_squished = old.combat_destroyed;
            const bool triggered = static_cast<size_t>(mower.row) < after.size() && after[mower.row] >= 0;
            if (!was_squished && !triggered) {
                ++events.mowers_used;
                events.mower_triggered_rows.push_back(mower.row);
            }
        }
    }

    events.waves_advanced = std::max(0, new_obs.global.current_wave - baseline_wave_);
    events.won = baseline_obs_.global.status != GameStatus::Won && new_status == GameStatus::Won;
    events.lost = baseline_obs_.global.status != GameStatus::Lost && new_status == GameStatus::Lost;

    reset();
    return events;
}

void EventsTracker::reset() {
    baseline_obs_ = Observation{};
    living_zombie_ids_.clear();
    living_plant_ids_.clear();
    baseline_sun_ = 0;
    baseline_wave_ = 0;
    has_baseline_ = false;
}

}  // 命名空间 pvz::env
