#include "reader_internal.h"
#include "../game/card_sources.h"

namespace pvz::observation {

bool read_cards_nolock(PvzRuntime::VersionedPvZ& pvz, uintptr_t board, const GlobalState& global, std::vector<CardState>& out_cards) {
    out_cards.clear();
    game::CardSources sources;
    if (!game::read_card_sources_nolock(pvz, board, global.sun, sources)) return false;
    for (const auto& source : sources.slots) {
        if (source.source == game::CardSourceKind::None) continue;
        CardState card;
        card.slot = source.slot;
        card.plant_type = source.plant_type;
        card.is_imitater = source.imitater;
        card.cost = source.cost;
        card.cooldown_duration = source.cooldown_duration;
        card.cooldown_remaining = source.cooldown_remaining;
        card.ready = source.ready;
        card.source = static_cast<int>(source.source);
        card.source_id = source.source_id;
        out_cards.push_back(card);
    }
    return true;
}

}  // namespace pvz::observation
