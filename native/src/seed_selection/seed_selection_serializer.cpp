#include "seed_selection_serializer.h"

#include <sstream>

namespace pvz::seed_selection {

std::string SeedSelectionSerializer::to_json(const SeedSelectionState& state) {
    std::stringstream ss;
    ss << "{\n";
    ss << "  \"present\": " << (state.present ? "true" : "false") << ",\n";
    ss << "  \"required_count\": " << state.required_count << ",\n";
    ss << "  \"selected_count\": " << state.selected_count << ",\n";
    ss << "  \"can_confirm\": " << (state.can_confirm ? "true" : "false") << ",\n";
    ss << "  \"seeds_in_flight\": " << (state.seeds_in_flight ? "true" : "false") << ",\n";
    ss << "  \"warning_flags\": " << state.warning_flags << ",\n";
    ss << "  \"imitater_target\": " << state.imitater_target << ",\n";
    ss << "  \"zombie_roster_available\": "
       << (state.zombie_roster_available ? "true" : "false") << ",\n";
    ss << "  \"zombie_types_in_level\": [";
    for (std::size_t i = 0; i < state.zombie_types_in_level.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << state.zombie_types_in_level[i];
    }
    ss << "],\n";

    ss << "  \"selected_seeds\": [";
    for (std::size_t i = 0; i < state.selected_seeds.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << state.selected_seeds[i];
    }
    ss << "],\n";

    ss << "  \"options\": [";
    for (std::size_t i = 0; i < state.options.size(); ++i) {
        const auto& option = state.options[i];
        if (i > 0) ss << ", ";
        ss << "{\"plant_type\": " << option.plant_type
           << ", \"available\": " << (option.available ? "true" : "false")
           << ", \"unlocked\": " << (option.unlocked ? "true" : "false")
           << ", \"selected\": " << (option.selected ? "true" : "false")
           << ", \"disabled\": " << (option.disabled ? "true" : "false")
           << ", \"chooser_state\": " << option.chooser_state << "}";
    }
    ss << "]\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace pvz::seed_selection
