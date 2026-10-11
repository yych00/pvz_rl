#include "card_sources.h"

#include "runtime/conveyor_seed_bank.h"

#include <algorithm>

namespace pvz::game {
namespace {

constexpr uintptr_t kPlantDefinitionBase = 0x0069F2C0;
constexpr uintptr_t kPlantDefinitionStride = 0x24;
constexpr int kFinalPlantCoinType = 7;
constexpr int kTrophyCoinType = 8;

const int kPlantCosts[] = {
    100, 50, 150, 50, 25, 175, 150, 200,
    0, 25, 75, 75, 75, 25, 75, 125,
    25, 50, 325, 25, 125, 100, 175, 125,
    0, 25, 125, 100, 125, 125, 125, 100,
    100, 25, 100, 75, 50, 100, 50, 300,
    250, 150, 150, 225, 200, 50, 125, 500,
    0
};

int plant_cost(PvzRuntime::VersionedPvZ& pvz, int type) {
    if (type >= 0 && type <= 48) {
        const uintptr_t addr = kPlantDefinitionBase +
            static_cast<uintptr_t>(type) * kPlantDefinitionStride;
        const int value = pvz.read_memory_checked<int>({addr + 0x10});
        if (value >= 0) return value;
    }
    if (type >= 0 && type < static_cast<int>(sizeof(kPlantCosts) / sizeof(kPlantCosts[0]))) {
        return kPlantCosts[type];
    }
    return 0;
}

int plant_cooldown(PvzRuntime::VersionedPvZ& pvz, int type) {
    if (type >= 0 && type <= 48) {
        const uintptr_t addr = kPlantDefinitionBase +
            static_cast<uintptr_t>(type) * kPlantDefinitionStride;
        const int value = pvz.read_memory_checked<int>({addr + 0x14});
        if (value > 0) return value;
    }
    if (type == 2 || type == 14 || type == 15 || type == 20 || (type >= 40 && type <= 47)) {
        return 5000;
    }
    if (type == 3 || type == 4 || type == 12 || type == 17 || type == 19 ||
        type == 23 || type == 24 || type == 25 || type == 30 || type == 38) {
        return 3000;
    }
    return 750;
}

int conveyor_packet_count_nolock(
    PvzRuntime::VersionedPvZ& pvz, uintptr_t seed_bank, int bank_slots) {
    for (int slot = 0; slot < bank_slots; ++slot) {
        const uintptr_t packet = seed_bank + static_cast<uintptr_t>(slot) * 0x50;
        if (pvz.read_memory_checked<int>({packet + pvz.data().slot_seed_type}) == -1) {
            return slot;
        }
    }
    return bank_slots;
}

CardSlotFact read_bank_slot_nolock(
    PvzRuntime::VersionedPvZ& pvz, uintptr_t seed_bank, bool conveyor,
    int sun, int slot) {
    CardSlotFact card;
    card.slot = slot;
    card.source = conveyor ? CardSourceKind::Conveyor : CardSourceKind::SeedBank;
    card.seed_bank = seed_bank;
    card.sun = sun;
    const uintptr_t base = seed_bank + static_cast<uintptr_t>(slot) * 0x50;
    card.raw_type = pvz.read_memory_checked<int>({base + pvz.data().slot_seed_type});
    if (conveyor && card.raw_type == -1) {
        card.source = CardSourceKind::None;
        card.gate = ActionError::CardUnavailable;
        return card;
    }
    const int imitater_type = pvz.read_memory_checked<int>({base + pvz.data().slot_seed_type_im});
    card.imitater = card.raw_type == 48;
    card.plant_type = card.imitater ? imitater_type : card.raw_type;
    const bool valid_type = card.plant_type >= 0 &&
        card.plant_type <= (conveyor ? 50 : 47);
    const bool active = pvz.read_memory_checked<bool>({base + 0x70});

    if (conveyor) {
        card.ready = valid_type && active &&
            runtime::conveyor_packet_visible_nolock(pvz, seed_bank, slot);
        card.gate = !valid_type ? ActionError::InvalidArgument :
            card.ready ? ActionError::None : ActionError::CardUnavailable;
        return card;
    }

    const int cd_past = pvz.read_memory_checked<int>({base + pvz.data().slot_seed_cd_past});
    int cd_total = pvz.read_memory_checked<int>({base + pvz.data().slot_seed_cd_total});
    const bool refreshing = pvz.read_memory_checked<bool>({base + 0x71});
    card.cooldown_on_use = plant_cooldown(pvz, card.plant_type);
    if (cd_total <= 0) cd_total = card.cooldown_on_use;
    card.cooldown_duration = cd_total;
    if ((!active || refreshing) && cd_total > cd_past) {
        card.cooldown_remaining = cd_total - cd_past;
    }
    card.cost = plant_cost(pvz, card.plant_type);
    card.ready = valid_type && card.cooldown_remaining <= 0 && sun >= card.cost;
    card.gate = !valid_type ? ActionError::InvalidArgument :
        ((!active || refreshing) && cd_past < cd_total) ? ActionError::CooldownActive :
        sun < card.cost ? ActionError::InsufficientSun : ActionError::None;
    return card;
}

CardSlotFact coin_slot(const CoinFact& coin, int slot) {
    CardSlotFact card;
    card.slot = slot;
    card.source = CardSourceKind::DroppedCoin;
    card.raw_type = coin.plant_type;
    card.plant_type = coin.plant_type;
    card.coin = coin.address;
    card.source_id = coin.id;
    card.ready = coin.collectable && coin.plant_type >= 0 && coin.plant_type <= 47;
    card.gate = !coin.collectable ? ActionError::CardUnavailable :
        (coin.plant_type < 0 || coin.plant_type > 47) ?
            ActionError::InvalidArgument : ActionError::None;
    return card;
}

void assign_seed_coin_slots(
    CardSources& sources, const std::vector<CoinFact>& coins) {
    int slot = sources.bank_slots;
    for (const auto& coin : coins) {
        if (coin.coin_type != kSeedCoinType) continue;
        if (slot >= kCardSlotCount) break;
        sources.slots[static_cast<std::size_t>(slot)] = coin_slot(coin, slot);
        ++slot;
    }
}

}  // namespace

bool read_relevant_coins_nolock(
    PvzRuntime::VersionedPvZ& pvz, uintptr_t board,
    std::vector<CoinFact>& out_coins) {
    out_coins.clear();
    if (!board) return false;
    const uintptr_t coin_array = pvz.read_memory_checked<uintptr_t>({
        board + PvzRuntime::coin_array_offset(pvz)});
    int count = pvz.read_memory_checked<int>({
        board + PvzRuntime::coin_array_offset(pvz) + 4});
    if (count < 0 || count > 512 || (count && !coin_array))
        throw PvzRuntime::NativeOperationError(ActionError::Unsupported, "invalid coin array");
    if (!coin_array || count == 0) return true;

    for (int i = 0; i < count; ++i) {
        const uintptr_t address = coin_array +
            static_cast<uintptr_t>(i) * PvzRuntime::kCoinStructSize;
        if (pvz.read_memory_checked<bool>({address + PvzRuntime::kCoinDeadOffset})) continue;
        const int coin_type = pvz.read_memory_checked<int>({
            address + PvzRuntime::kCoinTypeOffset});
        if (coin_type != kSeedCoinType && coin_type != kFinalPlantCoinType &&
            coin_type != kTrophyCoinType) continue;
        CoinFact coin;
        coin.id = i;
        coin.coin_type = coin_type;
        coin.plant_type = coin_type == kTrophyCoinType ? -1 :
            pvz.read_memory_checked<int>({address + PvzRuntime::kCoinPlantTypeOffset});
        coin.x = pvz.read_memory_checked<float>({address + 0x24});
        coin.y = pvz.read_memory_checked<float>({address + 0x28});
        coin.collectable = !pvz.read_memory_checked<bool>({
            address + PvzRuntime::kCoinBeingCollectedOffset});
        coin.address = address;
        out_coins.push_back(coin);
    }
    return true;
}

bool read_card_sources_nolock(
    PvzRuntime::VersionedPvZ& pvz, uintptr_t board, int sun,
    CardSources& out_sources) {
    out_sources = {};
    if (!board) return false;
    out_sources.bank_slots = runtime::seed_bank_slot_count_nolock(pvz, board);
    const uintptr_t seed_bank = pvz.read_memory_checked<uintptr_t>({board + pvz.data().slot});
    const bool conveyor = seed_bank &&
        runtime::has_conveyor_seed_bank_nolock(pvz, board);
    const int packet_slots = conveyor
        ? conveyor_packet_count_nolock(pvz, seed_bank, out_sources.bank_slots)
        : out_sources.bank_slots;
    for (int slot = 0; slot < kCardSlotCount; ++slot) {
        out_sources.slots[static_cast<std::size_t>(slot)].slot = slot;
        out_sources.slots[static_cast<std::size_t>(slot)].gate =
            ActionError::CardUnavailable;
    }
    for (int slot = 0; slot < packet_slots; ++slot) {
        out_sources.slots[static_cast<std::size_t>(slot)] =
            read_bank_slot_nolock(pvz, seed_bank, conveyor, sun, slot);
    }

    if (out_sources.bank_slots < kCardSlotCount) {
        std::vector<CoinFact> coins;
        if (!read_relevant_coins_nolock(pvz, board, coins)) return false;
        assign_seed_coin_slots(out_sources, coins);
    }
    return true;
}

bool read_card_source_nolock(
    PvzRuntime::VersionedPvZ& pvz, uintptr_t board, int sun, int slot,
    CardSlotFact& out_source) {
    out_source = {};
    out_source.slot = slot;
    if (!board || slot < 0 || slot >= kCardSlotCount) return false;
    const int bank_slots = runtime::seed_bank_slot_count_nolock(pvz, board);
    if (slot < bank_slots) {
        const uintptr_t seed_bank = pvz.read_memory_checked<uintptr_t>({board + pvz.data().slot});
        const bool conveyor = runtime::has_conveyor_seed_bank_nolock(pvz, board);
        if (conveyor && slot >= conveyor_packet_count_nolock(
                pvz, seed_bank, bank_slots)) {
            out_source.gate = ActionError::CardUnavailable;
            return true;
        }
        out_source = read_bank_slot_nolock(pvz, seed_bank, conveyor, sun, slot);
        return true;
    }
    std::vector<CoinFact> coins;
    if (!read_relevant_coins_nolock(pvz, board, coins)) return false;
    CardSources sources;
    sources.bank_slots = bank_slots;
    assign_seed_coin_slots(sources, coins);
    out_source = sources.slots[static_cast<std::size_t>(slot)];
    out_source.slot = slot;
    if (out_source.source == CardSourceKind::None) {
        out_source.gate = ActionError::CardUnavailable;
    }
    return true;
}

}  // namespace pvz::game
