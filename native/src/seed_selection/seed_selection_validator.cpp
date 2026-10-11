#include "seed_selection_validator.h"

#include <algorithm>

namespace pvz::seed_selection {
namespace {

const SeedOptionState* find_option(int plant_type, const SeedSelectionState& state) {
    auto it = std::find_if(state.options.begin(), state.options.end(),
        [plant_type](const SeedOptionState& option) { return option.plant_type == plant_type; });
    return it == state.options.end() ? nullptr : &(*it);
}

}  // namespace

SeedSelectionError SeedSelectionValidator::can_select(int plant_type, const SeedSelectionState& state) {
    if (!state.present) return SeedSelectionError::InvalidPhase;
    if (plant_type < 0 || plant_type >= kSeedTypeCapacity) return SeedSelectionError::InvalidArgument;
    if (plant_type == kImitaterType) return SeedSelectionError::InvalidArgument;
    if (state.seeds_in_flight) return SeedSelectionError::SeedsInFlight;
    if (state.required_count > 0 && state.selected_count >= state.required_count) {
        return SeedSelectionError::SeedSelectionFull;
    }

    const auto* option = find_option(plant_type, state);
    if (!option || !option->available || !option->unlocked || option->disabled) {
        return SeedSelectionError::SeedNotAvailable;
    }
    if (option->selected) return SeedSelectionError::SeedAlreadySelected;
    return SeedSelectionError::None;
}

SeedSelectionError SeedSelectionValidator::can_select_imitater(
    int target_type, const SeedSelectionState& state) {
    if (target_type < 0 || target_type >= kImitaterTargetCount) {
        return SeedSelectionError::InvalidArgument;
    }
    if (!state.present) return SeedSelectionError::InvalidPhase;
    if (state.seeds_in_flight) return SeedSelectionError::SeedsInFlight;
    if (state.required_count > 0 && state.selected_count >= state.required_count) {
        return SeedSelectionError::SeedSelectionFull;
    }
    const auto* imitater = find_option(kImitaterType, state);
    const auto* target = find_option(target_type, state);
    if (!imitater || !imitater->available || !imitater->unlocked || imitater->disabled ||
        !target || !target->unlocked || target->chooser_state == 4) {
        return SeedSelectionError::SeedNotAvailable;
    }
    if (imitater->selected) return SeedSelectionError::SeedAlreadySelected;
    return SeedSelectionError::None;
}

SeedSelectionError SeedSelectionValidator::can_deselect(int plant_type, const SeedSelectionState& state) {
    if (!state.present) return SeedSelectionError::InvalidPhase;
    if (plant_type < 0 || plant_type >= kSeedTypeCapacity) return SeedSelectionError::InvalidArgument;
    if (state.seeds_in_flight) return SeedSelectionError::SeedsInFlight;

    const auto* option = find_option(plant_type, state);
    if (!option || !option->selected) return SeedSelectionError::SeedNotSelected;
    // Dave 强制选中的植物 disabled=true，不允许玩家撤掉
    if (option->disabled) return SeedSelectionError::SeedNotAvailable;
    return SeedSelectionError::None;
}

SeedSelectionError SeedSelectionValidator::can_confirm(const SeedSelectionState& state) {
    if (!state.present) return SeedSelectionError::InvalidPhase;
    if (state.seeds_in_flight) return SeedSelectionError::SeedsInFlight;
    if (!state.can_confirm || state.required_count <= 0 || state.selected_count != state.required_count) {
        return SeedSelectionError::SeedSelectionIncomplete;
    }
    return SeedSelectionError::None;
}

std::vector<bool> SeedSelectionValidator::build_action_mask(
    const SeedSelectionState& state, int seed_type_capacity) {
    if (seed_type_capacity <= 0) return {};

    // The ABI has fixed 64-wide blocks. A different capacity must not shift IDs.
    if (seed_type_capacity != kSeedTypeCapacity) return {};
    std::vector<bool> mask(static_cast<std::size_t>(kSeedActionCount), false);
    for (int plant_type = 0; plant_type < seed_type_capacity; ++plant_type) {
        mask[static_cast<std::size_t>(kSeedSelectBase + plant_type)] =
            can_select(plant_type, state) == SeedSelectionError::None;
        mask[static_cast<std::size_t>(kSeedDeselectBase + plant_type)] =
            can_deselect(plant_type, state) == SeedSelectionError::None;
    }
    for (int target = 0; target < kImitaterTargetCount; ++target) {
        mask[static_cast<std::size_t>(kSeedImitaterBase + target)] =
            can_select_imitater(target, state) == SeedSelectionError::None;
    }
    mask[static_cast<std::size_t>(kSeedConfirmIndex)] =
        can_confirm(state) == SeedSelectionError::None;
    return mask;
}

}  // namespace pvz::seed_selection
