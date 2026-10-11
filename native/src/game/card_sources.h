#pragma once

#include "pvz/types.h"
#include "runtime/pvz_runtime_layout.h"

#include <array>
#include <cstdint>
#include <vector>

namespace pvz::game {

constexpr int kCardSlotCount = 10;
constexpr int kSeedCoinType = 16;

enum class CardSourceKind {
    None,
    SeedBank,
    Conveyor,
    DroppedCoin,
};

struct CoinFact {
    int id = -1;
    int coin_type = -1;
    int plant_type = -1;
    float x = 0.0f;
    float y = 0.0f;
    bool collectable = false;
    uintptr_t address = 0;
};

struct CardSlotFact {
    CardSourceKind source = CardSourceKind::None;
    int slot = -1;
    int raw_type = -1;
    int plant_type = -1;
    bool imitater = false;
    // Observation readiness and action validation are distinct facts.
    bool ready = false;
    int cost = 0;
    int cooldown_duration = 0;
    int cooldown_remaining = 0;
    int cooldown_on_use = 0;
    int sun = 0;
    uintptr_t seed_bank = 0;
    uintptr_t coin = 0;
    int source_id = -1;
    ActionError gate = ActionError::SlotIndexOutOfRange;
};

struct CardSources {
    // Bank slots keep their indices even when a conveyor packet is empty.
    int bank_slots = 0;
    std::array<CardSlotFact, kCardSlotCount> slots{};
};

// All readers require the process lock. Callers that need a consistent board
// snapshot must also hold their existing main-loop barrier.
bool read_relevant_coins_nolock(
    PvzRuntime::VersionedPvZ& pvz, uintptr_t board,
    std::vector<CoinFact>& out_coins);
bool read_card_sources_nolock(
    PvzRuntime::VersionedPvZ& pvz, uintptr_t board, int sun,
    CardSources& out_sources);
bool read_card_source_nolock(
    PvzRuntime::VersionedPvZ& pvz, uintptr_t board, int sun, int slot,
    CardSlotFact& out_source);

}  // namespace pvz::game
