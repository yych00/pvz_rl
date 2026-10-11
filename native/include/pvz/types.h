#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pvz {

enum class GameStatus : std::int32_t {
    Unknown = 0,
    Menu = 1,
    ChoosingSeeds = 2,
    Playing = 3,
    Won = 4,
    Lost = 5
};

enum class RngSeedSupport : std::int32_t {
    Unsupported = 0,
    Full = 2
};

enum class ActionType : std::int32_t {
    NoOp = 0,
    Plant = 1,
    Shovel = 2,
    CobFire = 3,
    InteractCell = 4
};

enum class ShovelTarget : std::int32_t {
    Normal = 0,
    Pumpkin = 1
};

enum class ActionError : std::int32_t {
    None = 0,
    GameNotFound = 1001,
    InvalidArgument = 1002,
    InsufficientSun = 1003,
    CooldownActive = 1004,
    SlotIndexOutOfRange = 1005,
    InvalidGridCoordinates = 1006,
    TerrainRequiresLilyPad = 1007,
    TerrainRequiresFlowerPot = 1008,
    GridBlockedByHazard = 1009,
    MainPlantAlreadyExists = 1010,
    UpgradeBaseMismatch = 1011,
    CobCannonDoubleBaseInvalid = 1012,
    PumpkinAlreadyEquipped = 1013,
    NoPlantToShovel = 1014,
    CobCannonRecharging = 1015,
    CobTargetOutOfArena = 1016,
    GameNotInBattleState = 1017,
    CobCannonNotFound = 1018,
    ActionTimeout = 1019,
    UnsupportedGameVersion = 1020,
    CardUnavailable = 1021,
    Unsupported = 1099
};

struct Action {
    ActionType type = ActionType::NoOp;
    std::int32_t card_slot = -1;
    std::int32_t row = -1;
    std::int32_t col = -1;
    ShovelTarget shovel_target = ShovelTarget::Normal;
    std::int32_t target_row = -1;
    std::int32_t target_col = -1;
};

struct ActionResult {
    bool success = false;
    ActionError error = ActionError::Unsupported;
    std::string message;
};

struct GlobalState {
    std::int64_t game_clock = 0;
    std::int32_t sun = 0;
    std::int32_t scene = 0;
    std::int32_t grid_rows = 5;
    std::int32_t level_id = 1;
    std::int32_t current_wave = 0;
    std::int32_t total_waves = 0;
    std::int32_t wave_countdown = 0;
    bool is_huge_wave = false;
    bool is_paused = false;
    GameStatus status = GameStatus::Unknown;
    std::vector<std::int32_t> zombie_types_in_level;
    // Full finite wave schedule, 50 entries per wave (unused entries are -1).
    // This is the actual Board::mZombiesInWave array, not the allowed-type list.
    std::vector<std::int32_t> zombie_spawn_plan;
    bool spawn_plan_available = false;
    std::int32_t game_mode = 0;
    std::string game_mode_name = "Adventure";
    std::string card_system = "seed_bank";
    bool is_conveyor = false;
    std::uint32_t board_id = 0; // Remote address; diagnostic identity only.
    std::int32_t raw_scene = -1;
    std::int32_t runtime_phase = -1;
    bool level_complete = false;
    // Board::mTriggeredLawnMowers counts StartMower, never victory cleanup.
    std::int32_t triggered_mowers = 0;
    std::vector<std::int32_t> mower_trigger_waves;
    // Board fog facts: column-major 9 x 7 alpha samples, including bottom fringe.
    bool fog_state_available = false;
    float fog_offset = 0.0f;
    std::vector<std::int32_t> fog_alpha;
};

struct CardState {
    std::int32_t slot = -1;
    std::int32_t plant_type = -1;
    std::int32_t cost = 0;
    std::int32_t cooldown_duration = 0;
    std::int32_t cooldown_remaining = 0;
    bool ready = false;
    // 1=seed bank, 2=conveyor, 3=dropped seed coin; dropped cards share coin ID.
    std::int32_t source = 0;
    std::int32_t source_id = -1;
    bool is_imitater = false; // plant_type remains the imitation target.
};

struct PlantState {
    std::int32_t id = -1;
    std::int32_t type = -1;
    std::int32_t row = -1;
    std::int32_t col = -1;
    std::int32_t hp = 0;
    std::int32_t max_hp = 0;
    // 同格多层植物用 layer 区分，例如睡莲/花盆、主体植物和南瓜。
    std::int32_t layer = 0;
    std::int32_t state = 0;
    bool is_sleeping = false;
    std::int32_t imitater_type = -1;
};

struct ZombieState {
    std::int32_t id = -1;            // 实例 ID；仅用于跨游戏刻追踪，不进模型。
    std::int32_t type = -1;          // 僵尸类型。

    // 行为状态机用于区分矿工出土、小丑爆炸准备、巨人投掷等特殊阶段。
    std::int32_t phase = 0;          // mZombiePhase：当前行为阶段。
    std::int32_t phase_counter = 0;  // mPhaseCounter：当前阶段计时。

    // 保留游戏原始空间和运动事实；C++ 层不做归一化。
    std::int32_t row = -1;           // 所在行。
    float x = 0.0f;                  // 原始横向坐标。
    float y = 0.0f;                  // 原始纵向坐标。
    float vel_x = 0.0f;              // mVelX；方向还受行为阶段、魅惑等影响，不等同于实际位移速度。
    float altitude = 0.0f;           // 连续离地高度；撑杆、小鬼、蹦极、气球等会使用。
    std::int32_t height = 0;         // 离散高度状态 mZombieHeight；与 altitude 不重复。

    // 独立耐久池用于保留路障/铁桶、铁门、气球等不同防御机制。
    std::int32_t body_hp = 0;
    std::int32_t max_hp = 0;
    std::int32_t helm_hp = 0;
    std::int32_t shield_hp = 0;
    std::int32_t flying_hp = 0;
    std::int32_t helm_max_hp = 0;
    std::int32_t shield_max_hp = 0;
    std::int32_t flying_max_hp = 0;
    std::int32_t from_wave = 0;
    bool is_dead_or_dying = false;

    // 控制效果保留剩余游戏刻，比单纯布尔值多保留“还剩多久”。
    std::int32_t chilled_counter = 0;   // 减速剩余时间。
    std::int32_t buttered_counter = 0;  // 黄油定身剩余时间。
    std::int32_t ice_trap_counter = 0;  // 冰封剩余时间。

    bool is_eating = false;             // 是否正在啃食植物。
    bool is_mind_controlled = false;    // 是否处于魅惑阵营。

    // 蹦极和僵王等特殊行为会使用显式目标；没有目标时为 -1。
    std::int32_t target_row = -1;
    std::int32_t target_col = -1;
};

struct DroppedCardState {
    std::int32_t id = -1;
    std::int32_t coin_type = -1;
    std::int32_t type = -1;
    float x = 0.0f;
    float y = 0.0f;
    bool is_collectable = false;
};

// Zomboss owns these attacks through a ReanimationID. They are neither
// ordinary Zombie instances nor entries in Board::mProjectiles.
struct BossProjectileState {
    std::uint32_t id = 0;
    std::int32_t owner_id = -1;
    std::int32_t type = -1; // 0=fireball, 1=iceball.
    std::int32_t row = -1;
    float x = 0.0f; // Reanimation::mOverlayMatrix.m02, continuous pixels.
    float y = 0.0f; // Reanimation::mOverlayMatrix.m12, continuous pixels.
    bool valid = false;
};

struct GridItemState {
    std::string type;
    std::int32_t row = -1;
    std::int32_t col = -1;
    std::int32_t remaining_ticks = 0;
};

struct IceTrailState {
    std::int32_t row = -1;
    std::int32_t col_start = -1;
    std::int32_t col_end = -1;
    std::int32_t melt_countdown = 0;
};

struct MowerState {
    std::int32_t row = -1;
    std::string state = "LOST";
    bool combat_destroyed = false; // SquishMower, including its dead slot.
};

struct GridSquareState {
    std::int32_t row = -1;
    std::int32_t col = -1;
    std::int32_t terrain = 0;
    bool has_base_plant = false;
    bool has_main_plant = false;
    bool has_pumpkin = false;
    bool has_grave = false;
    bool has_crater = false;
};

struct Observation {
    GlobalState global;
    std::vector<CardState> cards;
    std::vector<PlantState> plants;
    std::vector<ZombieState> zombies;
    // Includes dying/dead occupied slots for damage diagnostics; not model input.
    std::vector<ZombieState> zombie_health;
    std::vector<DroppedCardState> dropped_cards;
    std::vector<GridItemState> grid_items;
    std::vector<IceTrailState> ice_trails;
    std::vector<MowerState> mowers;
    std::vector<GridSquareState> grid;
    std::vector<BossProjectileState> boss_projectiles;
    std::uint32_t schema_version = 4;
    std::string quality = "fresh";
};

struct StepEvents {
    std::int32_t zombies_killed = 0;
    std::int32_t plants_lost = 0;
    std::int32_t sun_collected = 0;
    std::int32_t sun_spent = 0;
    std::int32_t mowers_used = 0;
    std::vector<std::int32_t> mower_triggered_rows;
    std::int32_t waves_advanced = 0;
    bool won = false;
    bool lost = false;
};

struct ResetConfig {
    std::int32_t level_id = 1;
    std::int32_t playthrough = 0; // 0=保持当前进度，1=首次冒险，2=重玩模式。
    std::uint64_t seed = 0;
    bool has_seed = false;
    bool stop_at_seed_selection = false;
    bool auto_select_seeds = true;
    std::vector<std::int32_t> seed_types;
    bool deterministic_pause = true;
    std::int32_t wait_ready_timeout_ms = 10000;
};

struct ResetResult {
    bool success = false;
    ActionError error = ActionError::Unsupported;
    std::string message;
    Observation observation;
    bool observation_available = false;
    GameStatus status = GameStatus::Unknown;
    std::int64_t game_clock = 0;
    RngSeedSupport rng_seed_support = RngSeedSupport::Unsupported;
    bool rng_seed_applied = false;
};

struct StepResult {
    bool success = false;
    ActionError error = ActionError::Unsupported;
    std::string message;
    Observation observation;
    StepEvents events;
    ActionResult action;
    std::vector<bool> action_mask;
    GameStatus status = GameStatus::Unknown;
    std::int64_t game_clock = 0;
    std::int32_t automation_collected = 0;
    bool terminal_observation_fallback = false;
    std::string observation_quality = "unavailable";
    std::int64_t last_confirmed_game_clock = 0;
    std::int64_t ticks_advanced = 0;
    bool ticks_advanced_exact = false;
    bool replay_eligible = false;
};

}  // 命名空间 pvz
