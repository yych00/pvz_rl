#pragma once

#include <cstdint>

#ifdef _WIN32
#define PVZ_GYM_API __declspec(dllexport)
#define PVZ_GYM_CALL __stdcall
#else
#define PVZ_GYM_API
#define PVZ_GYM_CALL
#endif

extern "C" {

// Runtime framework v1. Lifecycle/configuration status: 0 OK, 1 unsupported,
// 2 invalid argument, 3 invalid state. Query support before any mutation.
// PID=0 retains automatic attachment. executor: 0 legacy/1 resident;
// transport: 0 JSON/1 binary/2 shared memory; snapshot: 0 legacy/1 shared.
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_get_runtime_capabilities_json(char* out, int capacity);
// Frozen last Turbo failure on this OS thread. Read-only, resize-safe; {} if none.
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_get_runtime_error_json(char* out, int capacity);
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_configure_runtime(
    uint32_t process_id, int executor, int transport, int snapshot);
// Flags are the
// TurboFlag bits in pvz/runtime.h. Enter before reset; exit is idempotent.
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_begin_turbo_mode(uint32_t flags);
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_end_turbo_mode();
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_close_runtime();
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_set_process_audio_muted(uint32_t process_id, bool mute);
// Identical single-execution result-request ABI to other step methods.
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_env_step_turbo(
    int action_type, int card_slot, int row, int col,
    int shovel_target, int target_row, int target_col,
    int ticks, char* out, int capacity);
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_advance_phase(int phase, int updates, char* out, int capacity);
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_read_runtime_snapshot(char* out, int capacity);

// 1. 进程与关卡
PVZ_GYM_API bool PVZ_GYM_CALL pvz_gym_attach();
PVZ_GYM_API bool PVZ_GYM_CALL pvz_gym_is_game_on();
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_get_ui();
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_get_runtime_phase();
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_get_runtime_state_json(char* out_json_buf, int max_len);
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_set_adventure_playthrough(int playthrough);
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_select_adventure_level(int area, int sub_level);
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_select_adventure_level_id(int level_id);
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_get_current_adventure_level(int* area, int* sub_level, int* level_id, int* ui);
PVZ_GYM_API const char* PVZ_GYM_CALL pvz_gym_error_message(int code);

// 2. 模拟时钟
PVZ_GYM_API bool PVZ_GYM_CALL pvz_gym_pause();
PVZ_GYM_API bool PVZ_GYM_CALL pvz_gym_resume();
PVZ_GYM_API bool PVZ_GYM_CALL pvz_gym_is_paused();
PVZ_GYM_API bool PVZ_GYM_CALL pvz_gym_step_one_tick();
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_advance_ticks(int ticks);
PVZ_GYM_API int64_t PVZ_GYM_CALL pvz_gym_get_game_clock();
// 原版引擎帧时长控制；10 毫秒是正常的逻辑和渲染节奏。
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_get_frame_duration();
PVZ_GYM_API bool PVZ_GYM_CALL pvz_gym_set_frame_duration(int time_ms);
// 高速无头训练模式：决策间保持棋盘暂停，推进时短暂接管原生主循环并取消帧时长等待。
// 返回真实的 mMainCounter 增量，不强制要求与请求游戏刻完全一致。
PVZ_GYM_API bool PVZ_GYM_CALL pvz_gym_begin_headless_mode();
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_advance_ticks_headless(int ticks);
PVZ_GYM_API bool PVZ_GYM_CALL pvz_gym_end_headless_mode();
// 最近一次启动无头模式的诊断信息。
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_get_headless_begin_status();
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_get_headless_diagnostics_json(
    char* out_json_buf, int max_len);

// 3. 战斗动作
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_can_execute_action(
    int action_type, int card_slot, int row, int col,
    int shovel_target, int target_row, int target_col);
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_execute_action(
    int action_type, int card_slot, int row, int col,
    int shovel_target, int target_row, int target_col,
    char* out_msg_buf, int max_msg_len);
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_get_action_mask(
    uint8_t* out_mask_buf, int max_size, int* out_count);
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_get_plant_legality_compare_json(
    char* out_json_buf, int max_len);
PVZ_GYM_API void PVZ_GYM_CALL pvz_gym_reset_plant_legality_compare_stats();

// 4. 观测
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_get_observation_json(char* out_json_buf, int max_len);

// 5. Gym 回合与状态转移的权威接口。
// Start on the same OS thread as step/reset. On buffer overflow, only read the
// pending result using this ID; never repeat the mutating call. A new begin
// invalidates the preceding result. A successful read consumes the result.
PVZ_GYM_API void PVZ_GYM_CALL pvz_gym_set_step_profiling(bool enabled);
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_get_step_timing_json(char* out_json_buf, int max_len);
PVZ_GYM_API uint64_t PVZ_GYM_CALL pvz_gym_begin_result_request();
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_read_result_request(
    uint64_t request_id, char* out_json_buf, int max_len);
// 旧版 reset 继续导出用于兼容；正式 Python Gym 使用 v2。
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_env_reset(
    int level_id, uint64_t seed, bool has_seed, int timeout_ms,
    char* out_obs_json, int max_obs_len, int64_t* out_clock);

PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_env_reset_v2(
    int level_id,
    int playthrough,
    uint64_t seed,
    bool has_seed,
    bool stop_at_seed_selection,
    bool auto_select_seeds,
    const int32_t* seed_types,
    int seed_count,
    bool deterministic_pause,
    int timeout_ms,
    char* out_result_json,
    int max_len);

// 从已经准备好的选卡界面继续进入战斗，不重新选关或重开关卡，用于同一会话内从选卡阶段交接到战斗阶段。
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_env_wait_for_battle(
    bool deterministic_pause,
    int timeout_ms,
    char* out_result_json,
    int max_len);

PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_env_step(
    int action_type, int card_slot, int row, int col,
    int shovel_target, int target_row, int target_col,
    int ticks, char* out_result_json, int max_len);

// 与 pvz_gym_env_step 使用相同的权威 StepResult 契约，但要求无头时钟已经启动，
// 并在一次原生调用中完成整个状态转移。ticks 只是目标推进量，真实 game_clock 可能因教程或终局切换略有差异。
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_env_step_headless(
    int action_type, int card_slot, int row, int col,
    int shovel_target, int target_row, int target_col,
    int ticks, char* out_result_json, int max_len);

// 不暂停的实时状态转移。target_ticks 表示距离上次返回观测至少经过的游戏刻，
// 不是在动作到达后额外等待的固定时长；返回观测和 game_clock 始终反映真实流逝时间。
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_env_step_continuous(
    int action_type, int card_slot, int row, int col,
    int shovel_target, int target_row, int target_col,
    int target_ticks, char* out_result_json, int max_len);

// 0 表示不支持，2 表示完整支持；不支持的版本不会通过猜测偏移强行设置随机种子。
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_get_rng_seed_support();

// 6. 自动化辅助
PVZ_GYM_API void PVZ_GYM_CALL pvz_gym_set_automation_config(bool enabled, bool auto_dialogue, bool auto_collect);
PVZ_GYM_API void PVZ_GYM_CALL pvz_gym_get_automation_config(bool* out_enabled, bool* out_auto_dialogue, bool* out_auto_collect);
PVZ_GYM_API void PVZ_GYM_CALL pvz_gym_set_auto_whack(bool enable);
PVZ_GYM_API bool PVZ_GYM_CALL pvz_gym_get_auto_whack();
PVZ_GYM_API bool PVZ_GYM_CALL pvz_gym_handle_prebattle();
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_automation_collect();
PVZ_GYM_API bool PVZ_GYM_CALL pvz_gym_automation_whack();
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_automation_tick();
PVZ_GYM_API void PVZ_GYM_CALL pvz_gym_set_auto_skip_tutorial(bool enable);
PVZ_GYM_API bool PVZ_GYM_CALL pvz_gym_get_auto_skip_tutorial();
PVZ_GYM_API void PVZ_GYM_CALL pvz_gym_set_no_crazy_dave(bool enable);
PVZ_GYM_API bool PVZ_GYM_CALL pvz_gym_get_no_crazy_dave();
PVZ_GYM_API void PVZ_GYM_CALL pvz_gym_set_auto_dialogue(bool enable);
PVZ_GYM_API bool PVZ_GYM_CALL pvz_gym_get_auto_dialogue();
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_get_tutorial_state();
PVZ_GYM_API bool PVZ_GYM_CALL pvz_gym_skip_tutorial();

// 7. 窗口与调试辅助
PVZ_GYM_API bool PVZ_GYM_CALL pvz_gym_set_window_always_on_top(bool enable);
PVZ_GYM_API void PVZ_GYM_CALL pvz_gym_set_background_running(bool enable);
PVZ_GYM_API bool PVZ_GYM_CALL pvz_gym_set_cheat_keys(bool enable);
PVZ_GYM_API bool PVZ_GYM_CALL pvz_gym_get_cheat_keys();
PVZ_GYM_API bool PVZ_GYM_CALL pvz_gym_send_key(uint32_t vk_code);
PVZ_GYM_API bool PVZ_GYM_CALL pvz_gym_cancel_intro();
PVZ_GYM_API void PVZ_GYM_CALL pvz_gym_set_auto_cancel_intro(bool enable);
PVZ_GYM_API bool PVZ_GYM_CALL pvz_gym_get_auto_cancel_intro();
PVZ_GYM_API uintptr_t PVZ_GYM_CALL pvz_gym_get_dialog(int dialog_id);
PVZ_GYM_API bool PVZ_GYM_CALL pvz_gym_kill_dialog(int dialog_id);

// 8. 选卡动作域
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_get_seed_type_capacity();
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_get_seed_selection_json(char* out_json_buf, int max_len);
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_get_seed_selection_mask(
    uint8_t* out_mask_buf, int max_size, int* out_count);
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_select_seed(
    int plant_type, char* out_msg_buf, int max_msg_len);
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_select_imitater(
    int target_type, char* out_msg_buf, int max_msg_len);
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_deselect_seed(
    int plant_type, char* out_msg_buf, int max_msg_len);
PVZ_GYM_API int PVZ_GYM_CALL pvz_gym_confirm_seeds(
    char* out_msg_buf, int max_msg_len);

}
