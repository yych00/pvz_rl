#include "action_codec.h"

namespace pvz::action {

bool ActionCodec::decode(int index, Action& out_action) {
    if (index < 0 || index >= kActionSpaceSize) return false;

    out_action = Action{};
    if (index == kNoOpIndex) {
        out_action.type = ActionType::NoOp;
        return true;
    }

    if (index < kShovelBase) {
        int offset = index - kPlantBase;
        out_action.type = ActionType::Plant;
        out_action.card_slot = offset / (kMaxRows * kCols);
        offset %= (kMaxRows * kCols);
        out_action.row = offset / kCols;
        out_action.col = offset % kCols;
        return true;
    }

    if (index < kCobBase) {
        int offset = index - kShovelBase;
        const int target = offset / kShovelCells;
        offset %= kShovelCells;
        out_action.type = ActionType::Shovel;
        out_action.shovel_target = static_cast<ShovelTarget>(target);
        out_action.row = offset / kCols;
        out_action.col = offset % kCols;
        return true;
    }

    if (index < kInteractBase) {
        const int cell = index - kCobBase;
        out_action.type = ActionType::CobFire;
        out_action.target_row = cell / kCols;
        out_action.target_col = cell % kCols;
        return true;
    }
    const int cell = index - kInteractBase;
    out_action.type = ActionType::InteractCell;
    out_action.row = cell / kCols;
    out_action.col = cell % kCols;
    return true;
}

int ActionCodec::encode(const Action& action) {
    switch (action.type) {
    case ActionType::NoOp:
        return kNoOpIndex;
    case ActionType::Plant:
        if (action.card_slot < 0 || action.card_slot >= kMaxSlots ||
            action.row < 0 || action.row >= kMaxRows ||
            action.col < 0 || action.col >= kCols) {
            return -1;
        }
        return kPlantBase +
               ((action.card_slot * kMaxRows + action.row) * kCols + action.col);
    case ActionType::Shovel: {
        const int target = static_cast<int>(action.shovel_target);
        if (target < 0 || target >= kShovelTargetCount ||
            action.row < 0 || action.row >= kMaxRows ||
            action.col < 0 || action.col >= kCols) {
            return -1;
        }
        return kShovelBase + target * kShovelCells + action.row * kCols + action.col;
    }
    case ActionType::CobFire:
        if (action.row != -1 || action.col != -1 ||
            action.target_row < 0 || action.target_row >= kMaxRows ||
            action.target_col < 0 || action.target_col >= kCols) {
            return -1;
        }
        return kCobBase + action.target_row * kCols + action.target_col;
    case ActionType::InteractCell:
        if (action.row < 0 || action.row >= kMaxRows ||
            action.col < 0 || action.col >= kCols) return -1;
        return kInteractBase + action.row * kCols + action.col;
    default:
        return -1;
    }
}

}  // namespace pvz::action
