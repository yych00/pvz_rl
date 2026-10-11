#include "../src/action/action_codec.h"
#include "test_check.h"

namespace {

void test_boundaries() {
    using pvz::Action;
    using pvz::ActionType;
    using pvz::ShovelTarget;
    using pvz::action::ActionCodec;

    static_assert(ActionCodec::kActionSpaceSize == 757);
    static_assert(ActionCodec::kCobBase == 649);
    static_assert(ActionCodec::kCobCount == 54);

    Action action;
    CHECK(ActionCodec::decode(0, action));
    CHECK(action.type == ActionType::NoOp);

    CHECK(ActionCodec::decode(540, action));
    CHECK(action.type == ActionType::Plant);
    CHECK(action.card_slot == 9 && action.row == 5 && action.col == 8);

    CHECK(ActionCodec::decode(541, action));
    CHECK(action.type == ActionType::Shovel);
    CHECK(action.shovel_target == ShovelTarget::Normal);
    CHECK(action.row == 0 && action.col == 0);

    CHECK(ActionCodec::decode(648, action));
    CHECK(action.type == ActionType::Shovel);
    CHECK(action.shovel_target == ShovelTarget::Pumpkin);
    CHECK(action.row == 5 && action.col == 8);

    CHECK(ActionCodec::decode(649, action));
    CHECK(action.type == ActionType::CobFire);
    CHECK(action.target_row == 0 && action.target_col == 0);

    CHECK(ActionCodec::decode(702, action));
    CHECK(action.type == ActionType::CobFire);
    CHECK(action.target_row == 5 && action.target_col == 8);

    CHECK(ActionCodec::decode(703, action));
    CHECK(action.type == ActionType::InteractCell);
    CHECK(action.row == 0 && action.col == 0);

    CHECK(ActionCodec::decode(756, action));
    CHECK(action.type == ActionType::InteractCell);
    CHECK(action.row == 5 && action.col == 8);

    CHECK(!ActionCodec::decode(-1, action));
    CHECK(!ActionCodec::decode(757, action));
}

void test_complete_round_trip() {
    using pvz::Action;
    using pvz::action::ActionCodec;
    for (int index = 0; index < ActionCodec::kActionSpaceSize; ++index) {
        Action action;
        CHECK(ActionCodec::decode(index, action));
        CHECK(ActionCodec::encode(action) == index);
    }
}

void test_invalid_cob_coordinates() {
    using pvz::Action;
    using pvz::ActionType;
    using pvz::action::ActionCodec;

    Action action;
    action.type = ActionType::CobFire;
    action.row = 0;
    action.col = 0;
    action.target_row = 0;
    action.target_col = 0;
    CHECK(ActionCodec::encode(action) == -1);
    action.row = -1;
    action.col = -1;
    action.target_row = 6;
    action.target_col = 0;
    CHECK(ActionCodec::encode(action) == -1);
}

}  // namespace

int main() {
    test_boundaries();
    test_complete_round_trip();
    test_invalid_cob_coordinates();
    return 0;
}
