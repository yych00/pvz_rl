#include <climits>
#include "serializer.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>

namespace pvz::api {
namespace {

std::string json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (char ch : value) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += ch; break;
        }
    }
    return out;
}

}  // namespace

std::string Serializer::observation_to_json(const Observation& obs) {
    std::stringstream ss;
    ss.imbue(std::locale::classic());
    ss << std::setprecision(std::numeric_limits<float>::max_digits10);
    ss << "{\n";
    ss << "  \"global\": {\n";
    ss << "    \"game_clock\": " << obs.global.game_clock << ",\n";
    ss << "    \"sun\": " << obs.global.sun << ",\n";
    ss << "    \"scene\": " << obs.global.scene << ",\n";
    ss << "    \"current_wave\": " << obs.global.current_wave << ",\n";
    ss << "    \"total_waves\": " << obs.global.total_waves << ",\n";
    ss << "    \"grid_rows\": " << obs.global.grid_rows << ",\n";
    ss << "    \"level_id\": " << obs.global.level_id << ",\n";
    ss << "    \"wave_countdown\": " << obs.global.wave_countdown << ",\n";
    ss << "    \"is_huge_wave\": " << (obs.global.is_huge_wave ? "true" : "false") << ",\n";
    ss << "    \"is_paused\": " << (obs.global.is_paused ? "true" : "false") << ",\n";
    ss << "    \"status\": " << static_cast<int32_t>(obs.global.status) << ",\n";
    ss << "    \"game_mode\": " << obs.global.game_mode << ",\n";
    ss << "    \"game_mode_name\": \"" << json_escape(obs.global.game_mode_name) << "\",\n";
    ss << "    \"card_system\": \"" << json_escape(obs.global.card_system) << "\",\n";
    ss << "    \"is_conveyor\": " << (obs.global.is_conveyor ? "true" : "false") << ",\n";
    ss << "    \"board_id\": " << obs.global.board_id << ",\n";
    ss << "    \"raw_scene\": " << obs.global.raw_scene << ",\n";
    ss << "    \"runtime_phase\": " << obs.global.runtime_phase << ",\n";
    ss << "    \"level_complete\": " << (obs.global.level_complete ? "true" : "false") << ",\n";
    ss << "    \"triggered_mowers\": " << obs.global.triggered_mowers << ",\n";
    ss << "    \"fog_state_available\": " << (obs.global.fog_state_available ? "true" : "false") << ",\n";
    ss << "    \"fog_offset\": " << obs.global.fog_offset << ",\n";
    ss << "    \"fog_alpha\": [";
    for (size_t i = 0; i < obs.global.fog_alpha.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << obs.global.fog_alpha[i];
    }
    ss << "],\n";
    ss << "    \"mower_trigger_waves\": [";
    for (size_t i = 0; i < obs.global.mower_trigger_waves.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << obs.global.mower_trigger_waves[i];
    }
    ss << "],\n";
    ss << "    \"zombie_types_in_level\": [";
    for (size_t i = 0; i < obs.global.zombie_types_in_level.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << obs.global.zombie_types_in_level[i];
    }
    ss << "],\n    \"spawn_plan_available\": " << (obs.global.spawn_plan_available ? "true" : "false");
    ss << ",\n    \"zombie_spawn_plan\": [";
    for (size_t i = 0; i < obs.global.zombie_spawn_plan.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << obs.global.zombie_spawn_plan[i];
    }
    ss << "]\n  },\n";

    ss << "  \"cards\": [";
    for (size_t i = 0; i < obs.cards.size(); ++i) {
        const auto& c = obs.cards[i];
        if (i > 0) ss << ", ";
        ss << "\n    {\"slot\": " << c.slot
           << ", \"source\": " << c.source
           << ", \"source_id\": " << c.source_id
           << ", \"plant_type\": " << c.plant_type
           << ", \"is_imitater\": " << (c.is_imitater ? "true" : "false")
           << ", \"cost\": " << c.cost
           << ", \"cooldown_duration\": " << c.cooldown_duration
           << ", \"cooldown_remaining\": " << c.cooldown_remaining
           << ", \"ready\": " << (c.ready ? "true" : "false") << "}";
    }
    if (!obs.cards.empty()) ss << "\n  ";
    ss << "],\n";

    ss << "  \"plants\": [";
    for (size_t i = 0; i < obs.plants.size(); ++i) {
        const auto& p = obs.plants[i];
        if (i > 0) ss << ", ";
        ss << "\n    {\"id\": " << p.id
           << ", \"type\": " << p.type
           << ", \"imitater_type\": " << p.imitater_type
           << ", \"row\": " << p.row
           << ", \"col\": " << p.col
           << ", \"hp\": " << p.hp
           << ", \"max_hp\": " << p.max_hp
           << ", \"layer\": " << p.layer
           << ", \"state\": " << p.state
           << ", \"is_sleeping\": " << (p.is_sleeping ? "true" : "false") << "}";
    }
    if (!obs.plants.empty()) ss << "\n  ";
    ss << "],\n";

    ss << "  \"zombies\": [";
    for (size_t i = 0; i < obs.zombies.size(); ++i) {
        const auto& z = obs.zombies[i];
        if (i > 0) ss << ", ";
        ss << "\n    {\"id\": " << z.id
           << ", \"type\": " << z.type
           << ", \"phase\": " << z.phase
           << ", \"phase_counter\": " << z.phase_counter
           << ", \"row\": " << z.row
           << ", \"x\": " << z.x
           << ", \"y\": " << z.y
           << ", \"vel_x\": " << z.vel_x
           << ", \"altitude\": " << z.altitude
           << ", \"height\": " << z.height
           << ", \"body_hp\": " << z.body_hp
           << ", \"max_hp\": " << z.max_hp
           << ", \"helm_hp\": " << z.helm_hp
           << ", \"shield_hp\": " << z.shield_hp
           << ", \"flying_hp\": " << z.flying_hp
           << ", \"helm_max_hp\": " << z.helm_max_hp
           << ", \"shield_max_hp\": " << z.shield_max_hp
           << ", \"flying_max_hp\": " << z.flying_max_hp
           << ", \"from_wave\": " << z.from_wave
           << ", \"is_dead_or_dying\": " << (z.is_dead_or_dying ? "true" : "false")
           << ", \"chilled_counter\": " << z.chilled_counter
           << ", \"buttered_counter\": " << z.buttered_counter
           << ", \"ice_trap_counter\": " << z.ice_trap_counter
           << ", \"is_eating\": " << (z.is_eating ? "true" : "false")
           << ", \"is_mind_controlled\": " << (z.is_mind_controlled ? "true" : "false")
           << ", \"target_row\": " << z.target_row
           << ", \"target_col\": " << z.target_col << "}";
    }
    if (!obs.zombies.empty()) ss << "\n  ";
    ss << "],\n";

    ss << "  \"zombie_health\": [";
    for (size_t i = 0; i < obs.zombie_health.size(); ++i) {
        const auto& z = obs.zombie_health[i];
        if (i > 0) ss << ", ";
        ss << "{\"id\":" << z.id << ",\"type\":" << z.type
           << ",\"phase\":" << z.phase << ",\"from_wave\":" << z.from_wave
           << ",\"body_hp\":" << z.body_hp << ",\"max_hp\":" << z.max_hp
           << ",\"helm_hp\":" << z.helm_hp << ",\"helm_max_hp\":" << z.helm_max_hp
           << ",\"shield_hp\":" << z.shield_hp << ",\"shield_max_hp\":" << z.shield_max_hp
           << ",\"flying_hp\":" << z.flying_hp << ",\"flying_max_hp\":" << z.flying_max_hp
           << ",\"is_dead_or_dying\":" << (z.is_dead_or_dying ? "true" : "false")
           << ",\"is_mind_controlled\":" << (z.is_mind_controlled ? "true" : "false") << "}";
    }
    ss << "],\n";

    ss << "  \"dropped_cards\": [";
    for (size_t i = 0; i < obs.dropped_cards.size(); ++i) {
        const auto& d = obs.dropped_cards[i];
        if (i > 0) ss << ", ";
        ss << "\n    {\"id\": " << d.id
           << ", \"coin_type\": " << d.coin_type
           << ", \"type\": " << d.type
           << ", \"x\": " << d.x
           << ", \"y\": " << d.y
           << ", \"is_collectable\": " << (d.is_collectable ? "true" : "false") << "}";
    }
    if (!obs.dropped_cards.empty()) ss << "\n  ";
    ss << "],\n";

    ss << "  \"grid_items\": [";
    for (size_t i = 0; i < obs.grid_items.size(); ++i) {
        const auto& gi = obs.grid_items[i];
        if (i > 0) ss << ", ";
        ss << "\n    {\"type\": \"" << json_escape(gi.type) << "\""
           << ", \"row\": " << gi.row
           << ", \"col\": " << gi.col
           << ", \"remaining_ticks\": " << gi.remaining_ticks << "}";
    }
    if (!obs.grid_items.empty()) ss << "\n  ";
    ss << "],\n";

    ss << "  \"ice_trails\": [";
    for (size_t i = 0; i < obs.ice_trails.size(); ++i) {
        const auto& it = obs.ice_trails[i];
        if (i > 0) ss << ", ";
        ss << "\n    {\"row\": " << it.row
           << ", \"col_start\": " << it.col_start
           << ", \"col_end\": " << it.col_end
           << ", \"melt_countdown\": " << it.melt_countdown << "}";
    }
    if (!obs.ice_trails.empty()) ss << "\n  ";
    ss << "],\n";

    ss << "  \"mowers\": [";
    for (size_t i = 0; i < obs.mowers.size(); ++i) {
        const auto& m = obs.mowers[i];
        if (i > 0) ss << ", ";
        ss << "\n    {\"row\": " << m.row
           << ", \"state\": \"" << json_escape(m.state) << "\""
           << ", \"combat_destroyed\": " << (m.combat_destroyed ? "true" : "false") << "}";
    }
    if (!obs.mowers.empty()) ss << "\n  ";
    ss << "],\n";

    ss << "  \"grid\": [";
    for (size_t i = 0; i < obs.grid.size(); ++i) {
        const auto& g = obs.grid[i];
        if (i > 0) ss << ", ";
        ss << "\n    {\"row\": " << g.row
           << ", \"col\": " << g.col
           << ", \"terrain\": " << g.terrain
           << ", \"has_base_plant\": " << (g.has_base_plant ? "true" : "false")
           << ", \"has_main_plant\": " << (g.has_main_plant ? "true" : "false")
           << ", \"has_pumpkin\": " << (g.has_pumpkin ? "true" : "false")
           << ", \"has_grave\": " << (g.has_grave ? "true" : "false")
           << ", \"has_crater\": " << (g.has_crater ? "true" : "false") << "}";
    }
    if (!obs.grid.empty()) ss << "\n  ";
    ss << "],\n  \"boss_projectiles\": [";
    for (size_t i = 0; i < obs.boss_projectiles.size(); ++i) {
        const auto& p = obs.boss_projectiles[i];
        if (i > 0) ss << ", ";
        ss << "{\"id\":" << p.id << ",\"owner_id\":" << p.owner_id
           << ",\"type\":" << p.type << ",\"row\":" << p.row
           << ",\"x\":" << p.x << ",\"y\":" << p.y
           << ",\"valid\":" << (p.valid ? "true" : "false") << "}";
    }
    ss << "],\n  \"schema_version\": " << obs.schema_version
       << ",\n  \"quality\": \"" << json_escape(obs.quality) << "\"\n}\n";
    return ss.str();
}

std::string Serializer::reset_result_to_json(const ResetResult& res) {
    std::stringstream ss;
    ss << "{\n";
    ss << "  \"success\": " << (res.success ? "true" : "false") << ",\n";
    ss << "  \"error\": " << static_cast<int32_t>(res.error) << ",\n";
    ss << "  \"message\": \"" << json_escape(res.message) << "\",\n";
    ss << "  \"observation_available\": " << (res.observation_available ? "true" : "false") << ",\n";
    ss << "  \"observation\": ";
    if (res.observation_available) ss << observation_to_json(res.observation);
    else ss << "null";
    ss << ",\n";
    ss << "  \"status\": " << static_cast<int32_t>(res.status) << ",\n";
    ss << "  \"game_clock\": " << res.game_clock << ",\n";
    ss << "  \"rng_seed_support\": " << static_cast<int32_t>(res.rng_seed_support) << ",\n";
    ss << "  \"rng_seed_applied\": " << (res.rng_seed_applied ? "true" : "false") << "\n";
    ss << "}\n";
    return ss.str();
}

std::string Serializer::step_result_to_json(const StepResult& res) {
    std::stringstream ss;
    ss << "{\n";
    ss << "  \"success\": " << (res.success ? "true" : "false") << ",\n";
    ss << "  \"error\": " << static_cast<int32_t>(res.error) << ",\n";
    ss << "  \"message\": \"" << json_escape(res.message) << "\",\n";
    ss << "  \"observation\": " << observation_to_json(res.observation) << ",\n";
    ss << "  \"events\": {\n";
    ss << "    \"zombies_killed\": " << res.events.zombies_killed << ",\n";
    ss << "    \"plants_lost\": " << res.events.plants_lost << ",\n";
    ss << "    \"sun_collected\": " << res.events.sun_collected << ",\n";
    ss << "    \"sun_spent\": " << res.events.sun_spent << ",\n";
    ss << "    \"mowers_used\": " << res.events.mowers_used << ",\n";
    ss << "    \"mower_triggered_rows\": [";
    for (size_t i = 0; i < res.events.mower_triggered_rows.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << res.events.mower_triggered_rows[i];
    }
    ss << "],\n";
    ss << "    \"waves_advanced\": " << res.events.waves_advanced << ",\n";
    ss << "    \"won\": " << (res.events.won ? "true" : "false") << ",\n";
    ss << "    \"lost\": " << (res.events.lost ? "true" : "false") << "\n  },\n";
    ss << "  \"action\": {\n";
    ss << "    \"success\": " << (res.action.success ? "true" : "false") << ",\n";
    ss << "    \"error\": " << static_cast<int32_t>(res.action.error) << ",\n";
    ss << "    \"message\": \"" << json_escape(res.action.message) << "\"\n  },\n";
    ss << "  \"action_mask\": [";
    for (size_t i = 0; i < res.action_mask.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << (res.action_mask[i] ? "1" : "0");
    }
    ss << "],\n";
    ss << "  \"status\": " << static_cast<int32_t>(res.status) << ",\n";
    ss << "  \"game_clock\": " << res.game_clock << ",\n";
    ss << "  \"automation_collected\": " << res.automation_collected << ",\n";
    ss << "  \"terminal_observation_fallback\": "
       << (res.terminal_observation_fallback ? "true" : "false") << ",\n";
    ss << "  \"observation_quality\": \"" << json_escape(res.observation_quality) << "\",\n";
    ss << "  \"last_confirmed_game_clock\": " << res.last_confirmed_game_clock << ",\n";
    ss << "  \"ticks_advanced\": " << res.ticks_advanced << ",\n";
    ss << "  \"ticks_advanced_exact\": " << (res.ticks_advanced_exact ? "true" : "false") << ",\n";
    ss << "  \"replay_eligible\": " << (res.replay_eligible ? "true" : "false") << "\n";
    ss << "}\n";
    return ss.str();
}

int Serializer::copy_to_c_buffer(const std::string& src, char* out_buf, int max_len) {
    if (!out_buf || max_len <= 0) return -1;
    if (src.size() >= static_cast<std::size_t>(INT_MAX)) return -1;
    const int src_len = static_cast<int>(src.size());
    if (src_len + 1 > max_len) return -(src_len + 1);
    std::memcpy(out_buf, src.c_str(), src_len);
    out_buf[src_len] = '\0';
    return src_len;
}

}  // namespace pvz::api
