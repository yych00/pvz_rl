#include "reader_internal.h"
#include "../game/card_sources.h"

namespace pvz::observation {

bool read_dropped_cards_nolock(PvzRuntime::VersionedPvZ& pvz, uintptr_t board, std::vector<DroppedCardState>& out_dropped_cards) {
    out_dropped_cards.clear();
    std::vector<game::CoinFact> coins;
    if (!game::read_relevant_coins_nolock(pvz, board, coins)) return false;
    for (const auto& coin : coins) {
        DroppedCardState item;
        item.id = coin.id;
        item.coin_type = coin.coin_type;
        item.type = coin.plant_type;
        item.x = coin.x;
        item.y = coin.y;
        item.is_collectable = coin.collectable;
        out_dropped_cards.push_back(item);
    }
    return true;
}

}  // 命名空间 pvz::observation
