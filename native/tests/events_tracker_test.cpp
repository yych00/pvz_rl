#include "../src/env/events_tracker.h"
#include "test_check.h"

#include <cstdint>

namespace {

constexpr std::int32_t data_array_id(int generation, int slot) {
    return static_cast<std::int32_t>(
        (static_cast<std::uint32_t>(generation) << 16) |
        static_cast<std::uint32_t>(slot));
}

pvz::Observation base_observation() {
    pvz::Observation obs;
    obs.global.sun = 100;
    obs.global.current_wave = 1;
    obs.global.status = pvz::GameStatus::Playing;
    obs.global.board_id = 1234;
    obs.global.mower_trigger_waves.assign(6, -100);
    obs.cards.push_back({0, 0, 100, 100, 0, true});
    obs.plants.push_back({data_array_id(1001, 1), 0, 0, 0, 300, 300, 1, 0, false});
    pvz::ZombieState zombie;
    zombie.id = data_array_id(1001, 7);
    zombie.type = 0;
    zombie.row = 0;
    zombie.x = 700.0f;
    zombie.y = 100.0f;
    zombie.body_hp = 200;
    zombie.max_hp = 200;
    obs.zombies.push_back(zombie);
    obs.mowers.push_back({0, "READY"});
    return obs;
}

void test_normal_loss_kill_income_mower_wave_and_win() {
    pvz::env::EventsTracker tracker;
    auto before = base_observation();
    tracker.begin_step(before);

    auto after = before;
    after.global.sun = 125;
    after.global.current_wave = 2;
    after.global.status = pvz::GameStatus::Won;
    after.plants.clear();
    after.zombies.clear();
    after.mowers[0].state = "ROLLING";
    after.global.triggered_mowers = 1;
    after.global.mower_trigger_waves[0] = 1;

    const auto events = tracker.end_step(
        after, pvz::Action{}, {true, pvz::ActionError::None, "noop"}, pvz::GameStatus::Won);
    CHECK(events.zombies_killed == 1);
    CHECK(events.plants_lost == 1);
    CHECK(events.sun_collected == 25);
    CHECK(events.sun_spent == 0);
    CHECK(events.mowers_used == 1);
    CHECK(events.mower_triggered_rows == std::vector<std::int32_t>{0});
    CHECK(events.waves_advanced == 1);
    CHECK(events.won);
    CHECK(!events.lost);
}

void test_cleanup_before_won_is_not_mower_use() {
    auto before = base_observation();
    before.zombies.clear();
    auto after = before;
    after.mowers.clear();
    for (auto status : {pvz::GameStatus::Playing, pvz::GameStatus::Won}) {
        pvz::env::EventsTracker tracker;
        tracker.begin_step(before);
        auto events = tracker.end_step(after, {}, {true, pvz::ActionError::None, "noop"}, status);
        CHECK(events.mowers_used == 0);
        CHECK(events.mower_triggered_rows.empty());
    }
}

void test_trigger_and_removal_inside_one_step_and_counter_reset() {
    auto before = base_observation();
    auto after = before;
    after.mowers.clear(); // Never observed ROLLING.
    after.global.triggered_mowers = 2;
    after.global.mower_trigger_waves[0] = 1;
    after.global.mower_trigger_waves[1] = 1;
    pvz::env::EventsTracker tracker;
    tracker.begin_step(before);
    auto events = tracker.end_step(after, {}, {true, pvz::ActionError::None, "noop"}, pvz::GameStatus::Playing);
    CHECK(events.mowers_used == 2);
    CHECK(events.mower_triggered_rows.size() == 2);
    tracker.begin_step(after);
    events = tracker.end_step(before, {}, {true, pvz::ActionError::None, "noop"}, pvz::GameStatus::Playing);
    CHECK(events.mowers_used == 0);
    after.global.board_id++;
    tracker.begin_step(before);
    events = tracker.end_step(after, {}, {true, pvz::ActionError::None, "noop"}, pvz::GameStatus::Playing);
    CHECK(events.mowers_used == 0);
}

void test_boss_squish_counts_once_but_not_after_activation() {
    auto before = base_observation();
    auto after = before;
    after.mowers[0] = {0, "LOST", true}; // Also represents dead SQUISHED slot.
    pvz::env::EventsTracker tracker;
    tracker.begin_step(before);
    auto events = tracker.end_step(after, {}, {true, pvz::ActionError::None, "noop"}, pvz::GameStatus::Playing);
    CHECK(events.mowers_used == 1);
    CHECK(events.mower_triggered_rows == std::vector<std::int32_t>{0});
    tracker.begin_step(after);
    events = tracker.end_step(after, {}, {true, pvz::ActionError::None, "noop"}, pvz::GameStatus::Playing);
    CHECK(events.mowers_used == 0);
    before.global.triggered_mowers = after.global.triggered_mowers = 1;
    before.global.mower_trigger_waves[0] = after.global.mower_trigger_waves[0] = 1;
    tracker.begin_step(before);
    events = tracker.end_step(after, {}, {true, pvz::ActionError::None, "noop"}, pvz::GameStatus::Playing);
    CHECK(events.mowers_used == 0);
}

void test_successful_shovel_is_not_combat_loss() {
    pvz::env::EventsTracker tracker;
    auto before = base_observation();
    tracker.begin_step(before);
    auto after = before;
    after.plants.clear();

    pvz::Action action;
    action.type = pvz::ActionType::Shovel;
    action.row = 0;
    action.col = 0;
    const auto events = tracker.end_step(
        after, action, {true, pvz::ActionError::None, "shoveled"}, pvz::GameStatus::Playing);
    CHECK(events.plants_lost == 0);
}

void test_self_consuming_plant_is_not_combat_loss() {
    pvz::env::EventsTracker tracker;
    auto before = base_observation();
    before.plants[0].type = 2; // Cherry Bomb
    tracker.begin_step(before);
    auto after = before;
    after.plants.clear();
    const auto events = tracker.end_step(
        after, pvz::Action{}, {true, pvz::ActionError::None, "noop"}, pvz::GameStatus::Playing);
    CHECK(events.plants_lost == 0);
}

void test_plant_spend_and_income_are_separate() {
    pvz::env::EventsTracker tracker;
    auto before = base_observation();
    tracker.begin_step(before);
    auto after = before;
    after.global.sun = 50; // +50 collected and -100 plant cost.
    after.plants.push_back({data_array_id(1002, 2), 0, 0, 1, 300, 300, 1, 0, false});

    pvz::Action action;
    action.type = pvz::ActionType::Plant;
    action.card_slot = 0;
    action.row = 0;
    action.col = 1;
    const auto events = tracker.end_step(
        after, action, {true, pvz::ActionError::None, "planted"}, pvz::GameStatus::Playing);
    CHECK(events.sun_spent == 100);
    CHECK(events.sun_collected == 50);
}

void test_loss_does_not_relabel_terminal_cleanup_as_zombie_kills() {
    pvz::env::EventsTracker tracker;
    auto before = base_observation();
    tracker.begin_step(before);
    auto after = before;
    after.global.status = pvz::GameStatus::Lost;
    after.zombies.clear();
    const auto events = tracker.end_step(
        after, pvz::Action{}, {true, pvz::ActionError::None, "noop"}, pvz::GameStatus::Lost);
    CHECK(events.zombies_killed == 0);
    CHECK(events.lost);
}

void test_generation_id_detects_zombie_slot_reuse() {
    pvz::env::EventsTracker tracker;
    auto before = base_observation();
    const auto old_id = before.zombies[0].id;
    tracker.begin_step(before);

    auto after = before;
    // Same low 16-bit slot (7), but a later generation: the old zombie died
    // and PvZ reused its DataArray slot for a newly spawned zombie in the same
    // RL step. Slot-only IDs would incorrectly see {7}->{7} and miss the kill.
    after.zombies[0].id = data_array_id(1002, 7);
    after.zombies[0].x = 760.0f;
    CHECK(after.zombies[0].id != old_id);
    CHECK((static_cast<std::uint32_t>(after.zombies[0].id) & 0xFFFFu) == 7u);

    const auto events = tracker.end_step(
        after, pvz::Action{}, {true, pvz::ActionError::None, "noop"}, pvz::GameStatus::Playing);
    CHECK(events.zombies_killed == 1);
}

void test_generation_id_detects_plant_slot_reuse() {
    pvz::env::EventsTracker tracker;
    auto before = base_observation();
    const auto old_id = before.plants[0].id;
    tracker.begin_step(before);

    auto after = before;
    // Same physical DataArray slot (1), different generation. This must be
    // treated as the old plant disappearing and a different plant existing now.
    after.plants[0].id = data_array_id(1002, 1);
    after.plants[0].type = 3; // Wall-nut; not a self-consuming plant.
    CHECK(after.plants[0].id != old_id);
    CHECK((static_cast<std::uint32_t>(after.plants[0].id) & 0xFFFFu) == 1u);

    const auto events = tracker.end_step(
        after, pvz::Action{}, {true, pvz::ActionError::None, "noop"}, pvz::GameStatus::Playing);
    CHECK(events.plants_lost == 1);
}

}  // namespace

int main() {
    test_normal_loss_kill_income_mower_wave_and_win();
    test_cleanup_before_won_is_not_mower_use();
    test_trigger_and_removal_inside_one_step_and_counter_reset();
    test_boss_squish_counts_once_but_not_after_activation();
    test_successful_shovel_is_not_combat_loss();
    test_self_consuming_plant_is_not_combat_loss();
    test_plant_spend_and_income_are_separate();
    test_loss_does_not_relabel_terminal_cleanup_as_zombie_kills();
    test_generation_id_detects_zombie_slot_reuse();
    test_generation_id_detects_plant_slot_reuse();
    return 0;
}
