#include "../src/observation/zombie_decoder.h"
#include "../src/api/serializer.h"
#include "test_check.h"

#include <cstring>
#include <vector>

namespace {
using Bytes = std::vector<std::uint8_t>;

template <typename T>
void put(Bytes& bytes, std::size_t offset, T value) {
    CHECK(offset + sizeof(value) <= bytes.size());
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

Bytes fixture(std::size_t size = 0x15C) {
    Bytes bytes(size, 0);
    put(bytes, size - 4, std::uint32_t{0x81230007});
    put(bytes, 0x1C, 2);
    put(bytes, 0x24, 17);   // Digger; same positive mVelX before/after reversal.
    put(bytes, 0x28, 37);
    put(bytes, 0x68, 123);
    put(bytes, 0x2C, -12.5f);
    put(bytes, 0x30, 200.25f);
    put(bytes, 0x34, 0.5f);
    put(bytes, 0x84, -20.5f);
    put(bytes, 0x64, 9);
    put(bytes, 0xC8, 111);
    put(bytes, 0xCC, 270);
    put(bytes, 0xD0, 222);
    put(bytes, 0xD4, 999);  // Max health is not current health.
    put(bytes, 0xDC, 333);
    put(bytes, 0xE0, 1100);
    put(bytes, 0xE4, 444);
    put(bytes, 0xE8, 500);
    put(bytes, 0x6C, 8);
    put(bytes, 0xAC, 101);
    put(bytes, 0xB0, 202);
    put(bytes, 0xB4, 303);
    put(bytes, 0x50, std::uint8_t{0});
    put(bytes, 0x51, std::uint8_t{1});
    put(bytes, 0xB8, std::uint8_t{1});
    put(bytes, 0x80, 4);    // Deliberately stale target values for a non-targeting type.
    put(bytes, 0x130, 1);
    return bytes;
}

bool decode(const Bytes& bytes, pvz::ZombieState& out, int rows = 5) {
    return pvz::observation::decode_zombie(bytes.data(), bytes.size(), 7, rows, out);
}

void test_facts_and_identity() {
    for (const auto size : {0x15Cu, 0x168u}) {
        auto bytes = fixture(size);
        pvz::ZombieState z;
        CHECK(decode(bytes, z));
        CHECK(static_cast<std::uint32_t>(z.id) == 0x81230007u);
        CHECK(z.type == 17 && z.phase == 37 && z.phase_counter == 123);
        CHECK(z.row == 2 && z.x == -12.5f && z.y == 200.25f);
        CHECK(z.vel_x == 0.5f && z.altitude == -20.5f && z.height == 9);
        CHECK(z.body_hp == 111 && z.max_hp == 270);
        CHECK(z.helm_hp == 222 && z.shield_hp == 333 && z.flying_hp == 444);
        CHECK(z.helm_max_hp == 999 && z.shield_max_hp == 1100 && z.flying_max_hp == 500);
        CHECK(z.from_wave == 8 && !z.is_dead_or_dying);
        CHECK(z.chilled_counter == 101 && z.buttered_counter == 202 && z.ice_trap_counter == 303);
        CHECK(z.is_eating && z.is_mind_controlled);
        CHECK(z.target_row == -1 && z.target_col == -1);
        put(bytes, size - 4, std::uint32_t{0x81240007});
        pvz::ZombieState replacement;
        CHECK(decode(bytes, replacement) && replacement.id != z.id);
        put(bytes, size - 4, std::uint32_t{7});
        CHECK(!decode(bytes, z)); // Freed slot.
        put(bytes, size - 4, std::uint32_t{0x81230008});
        CHECK(!decode(bytes, z)); // Wrong slot.
    }
}

void test_lifecycle_and_raw_values() {
    pvz::ZombieState z;
    for (int phase : {1, 2, 3}) {
        auto bytes = fixture();
        put(bytes, 0x28, phase);
        CHECK(!decode(bytes, z)); // Still has HP, but no longer a living threat.
        CHECK(pvz::observation::decode_zombie(bytes.data(), bytes.size(), 7, 5, z, true));
        CHECK(z.is_dead_or_dying && z.body_hp == 111);
    }
    auto bytes = fixture();
    put(bytes, 0xEC, std::uint8_t{1});
    CHECK(!decode(bytes, z));
    CHECK(pvz::observation::decode_zombie(bytes.data(), bytes.size(), 7, 5, z, true));
    CHECK(z.is_dead_or_dying);
    put(bytes, 0xEC, std::uint8_t{0});
    put(bytes, 0xC8, -10);
    put(bytes, 0xD0, -20);
    CHECK(decode(bytes, z) && z.body_hp == -10 && z.helm_hp == -20);
    put(bytes, 0x1C, 5);
    CHECK(!decode(bytes, z, 5));
    CHECK(decode(bytes, z, 6));
    put(bytes, 0x24, 25);
    put(bytes, 0x1C, -1);
    CHECK(decode(bytes, z) && z.row == -1); // Do not invent a Boss row.
    CHECK(!decode(Bytes(0x160), z)); // Unknown/beta layout.
    CHECK(!decode(Bytes(16), z));
    CHECK(!pvz::observation::decode_zombie(nullptr, 0x15C, 7, 5, z));
}

void test_targets() {
    auto bytes = fixture();
    pvz::ZombieState z;
    put(bytes, 0x24, 20);
    for (int phase = 4; phase <= 8; ++phase) {
        put(bytes, 0x28, phase);
        CHECK(decode(bytes, z) && z.target_row == 2 && z.target_col == 4);
    }
    for (int col : {-1, 9}) {
        put(bytes, 0x80, col);
        CHECK(decode(bytes, z) && z.target_row == -1 && z.target_col == -1);
    }
    put(bytes, 0x80, 4);
    for (int phase : {0, 9, 10}) {
        put(bytes, 0x28, phase);
        CHECK(decode(bytes, z) && z.target_row == -1 && z.target_col == -1);
    }
    put(bytes, 0x24, 25);
    for (int phase : {80, 81}) {
        put(bytes, 0x28, phase);
        CHECK(decode(bytes, z) && z.target_row == 1 && z.target_col == -1);
    }
    for (int phase : {82, 83, 84}) {
        put(bytes, 0x28, phase);
        CHECK(decode(bytes, z) && z.target_row == -1 && z.target_col == 4);
    }
    put(bytes, 0x28, 85);
    CHECK(decode(bytes, z) && z.target_row == 1 && z.target_col == 4);
    put(bytes, 0x28, 79); // Old attack coordinates survive in memory during idle.
    CHECK(decode(bytes, z) && z.target_row == -1 && z.target_col == -1);
}

std::string test_json() {
    pvz::Observation obs;
    pvz::ZombieState z;
    CHECK(decode(fixture(), z));
    obs.zombies.push_back(z);
    obs.zombie_health.push_back(z);
    obs.global.spawn_plan_available = true;
    obs.global.zombie_spawn_plan = {0, 2, -1};
    pvz::CardState card;
    card.source = 3;
    card.source_id = 17;
    obs.cards.push_back(card);
    card.is_imitater = true;
    card.plant_type = 14;
    obs.cards.push_back(card);
    pvz::PlantState plant;
    plant.type = 48;
    plant.imitater_type = 2;
    obs.plants.push_back(plant);
    const auto json = pvz::api::Serializer::observation_to_json(obs);
    CHECK(json.find("\"is_imitater\": false") != std::string::npos);
    CHECK(json.find("\"plant_type\": 14, \"is_imitater\": true") != std::string::npos);
    for (const char* fact : {"\"zombie_health\": [", "\"helm_max_hp\":999",
            "\"shield_max_hp\":1100", "\"flying_max_hp\":500", "\"from_wave\":8",
            "\"spawn_plan_available\": true", "\"zombie_spawn_plan\": [0, 2, -1]",
            "\"source\": 3", "\"source_id\": 17", "\"imitater_type\": 2"}) {
        CHECK(json.find(fact) != std::string::npos);
    }
    for (const char* fact : {"\"body_hp\": 111", "\"helm_hp\": 222", "\"shield_hp\": 333",
             "\"flying_hp\": 444", "\"phase\": 37", "\"phase_counter\": 123",
             "\"vel_x\": 0.5", "\"height\": 9", "\"altitude\": -20.5",
             "\"chilled_counter\": 101", "\"buttered_counter\": 202", "\"ice_trap_counter\": 303",
             "\"is_eating\": true", "\"is_mind_controlled\": true",
             "\"target_row\": -1", "\"target_col\": -1"}) {
        CHECK(json.find(fact) != std::string::npos);
    }
    for (const char* old : {"\"is_frozen\":", "\"is_slowed\":"}) {
        CHECK(json.find(old) == std::string::npos);
    }
    // A six-significant-digit stream default would silently round this fact.
    obs.zombies[0].vel_x = 0.123456789f;
    const auto precise = pvz::api::Serializer::observation_to_json(obs);
    const std::string key = "\"vel_x\": ";
    const auto start = precise.find(key);
    CHECK(start != std::string::npos);
    CHECK(std::stof(precise.substr(start + key.size())) == obs.zombies[0].vel_x);
    return json;
}
}  // namespace

int main(int argc, char**) {
    test_facts_and_identity();
    test_lifecycle_and_raw_values();
    test_targets();
    const auto json = test_json();
    if (argc > 1) std::cout << json;
    return 0;
}
