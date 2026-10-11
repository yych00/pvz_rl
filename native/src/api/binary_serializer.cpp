#include "binary_serializer.h"
namespace pvz::api {
#define START(TYPE) void BinaryWriter::value(const TYPE& v) { auto at = begin_map(); std::uint32_t count = 0;
#define F(NAME) field(#NAME, v.NAME, count);
#define END end_map(at, count); }
START(GlobalState)
F(game_clock) F(sun) F(scene) F(grid_rows) F(level_id) F(current_wave) F(total_waves)
F(wave_countdown) F(is_huge_wave) F(is_paused) F(status) F(zombie_types_in_level)
F(zombie_spawn_plan) F(spawn_plan_available) F(game_mode) F(game_mode_name) F(card_system) F(is_conveyor)
F(board_id) F(raw_scene) F(runtime_phase) F(level_complete) F(triggered_mowers) F(mower_trigger_waves)
F(fog_state_available) F(fog_offset) F(fog_alpha)
END
START(CardState) F(slot) F(plant_type) F(cost) F(cooldown_duration) F(cooldown_remaining) F(ready) F(source) F(source_id) F(is_imitater) END
START(PlantState) F(id) F(type) F(row) F(col) F(hp) F(max_hp) F(layer) F(state) F(is_sleeping) F(imitater_type) END
START(ZombieState)
F(id) F(type) F(phase) F(phase_counter) F(row) F(x) F(y) F(vel_x) F(altitude) F(height)
F(body_hp) F(max_hp) F(helm_hp) F(shield_hp) F(flying_hp) F(helm_max_hp) F(shield_max_hp)
F(flying_max_hp) F(from_wave) F(is_dead_or_dying) F(chilled_counter) F(buttered_counter)
F(ice_trap_counter) F(is_eating) F(is_mind_controlled) F(target_row) F(target_col)
END
START(BossProjectileState) F(id) F(owner_id) F(type) F(row) F(x) F(y) F(valid) END
START(DroppedCardState) F(id) F(coin_type) F(type) F(x) F(y) F(is_collectable) END
START(GridItemState) F(type) F(row) F(col) F(remaining_ticks) END
START(IceTrailState) F(row) F(col_start) F(col_end) F(melt_countdown) END
START(MowerState) F(row) F(state) F(combat_destroyed) END
START(GridSquareState) F(row) F(col) F(terrain) F(has_base_plant) F(has_main_plant) F(has_pumpkin) F(has_grave) F(has_crater) END
START(Observation) F(global) F(cards) F(plants) F(zombies) F(zombie_health) F(dropped_cards) F(grid_items) F(ice_trails) F(mowers) F(grid) F(boss_projectiles) F(schema_version) F(quality) END
START(StepEvents) F(zombies_killed) F(plants_lost) F(sun_collected) F(sun_spent) F(mowers_used) F(mower_triggered_rows) F(waves_advanced) F(won) F(lost) END
START(ActionResult) F(success) F(error) F(message) END
START(ResetResult)
F(success) F(error) F(message) F(observation_available)
value("observation"); if (v.observation_available) value(v.observation); else null(); ++count;
F(status) F(game_clock) F(rng_seed_support) F(rng_seed_applied)
END
START(StepResult)
F(success) F(error) F(message) F(observation) F(events) F(action) F(action_mask) F(status)
F(game_clock) F(automation_collected) F(terminal_observation_fallback)
F(observation_quality) F(last_confirmed_game_clock) F(ticks_advanced) F(ticks_advanced_exact) F(replay_eligible)
END
#undef START
#undef F
#undef END
} // namespace pvz::api
