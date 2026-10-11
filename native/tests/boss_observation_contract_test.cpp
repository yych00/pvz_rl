#include "../src/observation/boss_projectile_snapshot.h"
#include "../src/env/transition_observation.h"
#include "../src/env/events_tracker.h"
#include "../src/api/serializer.h"
#include "../src/api/binary_serializer.h"
#include "test_check.h"

#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>

template<class T> void put(std::vector<std::uint8_t>& bytes, std::size_t offset, T value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

int main(int argc, char** argv) {
    using namespace pvz;
    using namespace pvz::observation;
    std::vector<std::uint8_t> boss(0x15C), reanim(0xA0);
    put(boss, 0x140, std::uint32_t(0x12340002));
    put(boss, 0x148, 3); boss[0x14C] = 1;
    put(reanim, 0, 0x54); put(reanim, 0x9C, std::uint32_t(0x12340002));
    put(reanim, 0x2C, 455.5f); put(reanim, 0x38, 230.25f);
    std::unordered_map<std::uintptr_t, std::vector<std::uint8_t>> memory;
    const auto scalar = [&](std::uintptr_t address, std::uint32_t value) {
        auto& bytes = memory[address]; bytes.resize(4); put(bytes, 0, value);
    };
    scalar(0x1000, 0x2000); scalar(0x2008, 0x3000);
    scalar(0x3000, 0x4000); scalar(0x3008, 4);
    int reads = 0, fail = 0, raise = 0;
    const auto read = [&](std::uintptr_t address, void* output, std::size_t size) {
        ++reads;
        if (reads == raise) throw std::runtime_error("injected read failure");
        if (reads == fail) return false;
        if (address == 0x4140 && size == reanim.size()) {
            std::memcpy(output, reanim.data(), size); return true;
        }
        const auto found = memory.find(address);
        if (found == memory.end() || found->second.size() != size) return false;
        std::memcpy(output, found->second.data(), size); return true;
    };
    std::vector<BossProjectileState> entities;
    const auto sample = [&] {
        entities.clear(); reads = 0;
        return read_boss_projectile_snapshot(read, 0x1000, boss.data(), boss.size(), 5, 42, entities);
    };
    CHECK(sample() && entities.size() == 1 && reads == 5);
    CHECK(entities[0].valid && entities[0].type == 0 && entities[0].row == 3);
    CHECK(entities[0].id == 0x12340002 && entities[0].owner_id == 42);
    CHECK(entities[0].x == 455.5f && entities[0].y == 230.25f);
    const auto fire = entities[0];
    for (fail = 1; fail <= 5; ++fail) { CHECK(!sample()); CHECK(entities.empty()); }
    fail = 0; raise = 3;
    bool threw = false;
    try { sample(); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw && entities.empty()); raise = 0;
    put(boss, 0x140, std::uint32_t(0));
    CHECK(sample() && entities.empty() && reads == 0);
    put(boss, 0x140, std::uint32_t(0x43210002));
    CHECK(sample() && entities.empty()); // Recycled generation, not a live ball.
    put(boss, 0x140, std::uint32_t(0x12340002)); reanim[0x14] = 1;
    CHECK(sample() && entities.empty()); reanim[0x14] = 0;
    put(boss, 0x148, 5); CHECK(!sample()); put(boss, 0x148, 3);
    put(reanim, 0x2C, std::numeric_limits<float>::quiet_NaN()); CHECK(!sample());
    put(reanim, 0x2C, -179.75f);
    boss[0x14C] = 0; put(reanim, 0, 0x53);
    CHECK(sample() && entities[0].type == 1 && entities[0].x == -179.75f);

    Observation before;
    before.global.board_id = 0x10000; before.global.game_clock = 100;
    before.global.status = GameStatus::Playing; before.global.sun = 150;
    before.global.scene = 3; before.global.grid_rows = 6;
    before.global.fog_state_available = true;
    before.global.fog_offset = 80.5f;
    before.global.fog_alpha.assign(63, 255);
    before.cards.push_back({0, 0, 100, 100, 0, true}); before.cards[0].is_imitater = true;
    before.boss_projectiles = {fire, entities[0]};
    auto after = before;
    after.global.game_clock = 109; after.global.status = GameStatus::Won;
    after.global.raw_scene = 3; after.global.level_complete = true;
    after.global.runtime_phase = static_cast<int>(game::RuntimePhase::LevelComplete);
    StepResult fresh;
    CHECK(env::prepare_transition_observation(fresh, before, after, true, {}, 100));
    CHECK(fresh.status == GameStatus::Won && fresh.ticks_advanced == 9);
    CHECK(fresh.ticks_advanced_exact && fresh.replay_eligible && fresh.observation_quality == "fresh");
    fresh.success = true; fresh.error = ActionError::None; fresh.observation = after;
    const auto fixture = fresh;
    after.global.board_id += 4;
    CHECK(!env::prepare_transition_observation(fresh, before, after, true, {}, 100));
    after = before; after.global.game_clock = 99;
    CHECK(!env::prepare_transition_observation(fresh, before, after, true, {}, 100));

    game::RuntimeState terminal;
    terminal.process_available = true; terminal.raw_scene = 5;
    terminal.phase = game::RuntimePhase::Award; terminal.status = GameStatus::Won;
    StepResult fallback;
    CHECK(env::prepare_transition_observation(fallback, before, after, false, terminal, 107));
    CHECK(fallback.status == after.global.status && fallback.status == GameStatus::Won);
    CHECK(after.global.game_clock == 100 && fallback.last_confirmed_game_clock == 107);
    CHECK(fallback.ticks_advanced == 7 && !fallback.ticks_advanced_exact && !fallback.replay_eligible);
    CHECK(after.quality == "terminal_fallback");
    terminal.board_present = true;
    CHECK(!env::prepare_transition_observation(fallback, before, after, false, terminal, 107));
    terminal.board_present = false; terminal.process_available = false;
    CHECK(!env::prepare_transition_observation(fallback, before, after, false, terminal, 107));

    env::EventsTracker tracker;
    tracker.begin_step(before);
    Action plant; plant.type = ActionType::Plant; plant.card_slot = 0;
    const auto events = tracker.end_step(after, plant, {true, ActionError::None, "planted"}, GameStatus::Won);
    CHECK(events.won && events.sun_collected == 0 && events.sun_spent == 0);
    CHECK(!tracker.end_step(after, plant, {true, ActionError::None, "planted"}, GameStatus::Won).won);
    tracker.begin_step(after);
    CHECK(!tracker.end_step(after, plant, {true, ActionError::None, "planted"}, GameStatus::Won).won);

    const auto json = api::Serializer::step_result_to_json(fixture);
    const auto binary = api::binary_result(fixture);
    CHECK(json.find("\"boss_projectiles\"") != std::string::npos);
    CHECK(json.find("\"is_imitater\": true") != std::string::npos);
    CHECK(binary.find("boss_projectiles") != std::string::npos);
    CHECK(binary.find("ticks_advanced_exact") != std::string::npos);
    CHECK(json.find("\"fog_state_available\": true") != std::string::npos);
    CHECK(json.find("\"fog_offset\": 80.5") != std::string::npos);
    CHECK(binary.find("fog_alpha") != std::string::npos);
    if (argc == 2) {
        // Optional fixtures let Python validate the actual C++ JSON/PVZB
        // serializers and shared-memory decoder together, without a game.
        std::ofstream(std::string(argv[1]) + ".json", std::ios::binary) << json;
        std::ofstream(std::string(argv[1]) + ".pvzb", std::ios::binary) << binary;
    }
}
