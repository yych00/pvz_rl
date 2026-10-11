#include "runtime_phase.h"

#include "../process/process_manager.h"

#include <mutex>

namespace pvz::game {
namespace {

constexpr std::uintptr_t kCutsceneSeedChoosingOffset = 0x2C;
constexpr std::uintptr_t kCutsceneTimeOffset = 0x08;
constexpr std::uintptr_t kCutsceneCrazyDaveTimeOffset = 0x20;
constexpr std::uintptr_t kBoardLevelCompleteOffset = 0x55FC;

// PvZ 1.2.0.1065 EN original layout:
// LawnApp::mPlayerInfo = LawnApp + 0x82C
// PlayerInfo::mFinishedAdventure = PlayerInfo + 0x2C
constexpr std::uintptr_t kLawnPlayerInfoOffset = 0x82C;
constexpr std::uintptr_t kPlayerFinishedAdventureOffset = 0x2C;

constexpr int kTimePanRightEnd = 3500;
constexpr int kAdventureGameMode = 0;

}  // namespace

RuntimePhase RuntimePhaseReader::classify(
    std::int32_t raw_scene,
    bool seed_choosing,
    bool dave_talking,
    bool level_complete,
    std::int32_t game_mode) {
    if (raw_scene == 3 && game_mode == kAdventureGameMode && level_complete) {
        return RuntimePhase::LevelComplete;
    }

    switch (raw_scene) {
        case 0: return RuntimePhase::Loading;       // SCENE_LOADING
        case 1: return RuntimePhase::Menu;          // SCENE_MENU
        case 2:                                      // SCENE_LEVEL_INTRO
            if (seed_choosing) return RuntimePhase::SeedSelection;
            if (dave_talking) return RuntimePhase::Dialogue;
            return RuntimePhase::LevelIntro;
        case 3: return RuntimePhase::Battle;        // SCENE_PLAYING
        case 4: return RuntimePhase::Lost;          // SCENE_ZOMBIES_WON
        case 5: return RuntimePhase::Award;         // SCENE_AWARD
        case 6: return RuntimePhase::Credit;        // SCENE_CREDIT
        case 7: return RuntimePhase::Challenge;     // SCENE_CHALLENGE
        default: return RuntimePhase::Unknown;
    }
}

GameStatus RuntimePhaseReader::to_game_status(RuntimePhase phase) {
    switch (phase) {
        case RuntimePhase::Menu:
            return GameStatus::Menu;
        case RuntimePhase::SeedSelection:
            return GameStatus::ChoosingSeeds;
        case RuntimePhase::Battle:
            return GameStatus::Playing;
        case RuntimePhase::Award:
        case RuntimePhase::LevelComplete:
            return GameStatus::Won;
        case RuntimePhase::Lost:
            return GameStatus::Lost;
        default:
            return GameStatus::Unknown;
    }
}

const char* RuntimePhaseReader::phase_name(RuntimePhase phase) {
    switch (phase) {
        case RuntimePhase::Offline: return "OFFLINE";
        case RuntimePhase::Loading: return "LOADING";
        case RuntimePhase::Menu: return "MENU";
        case RuntimePhase::LevelIntro: return "LEVEL_INTRO";
        case RuntimePhase::Dialogue: return "DIALOGUE";
        case RuntimePhase::SeedSelection: return "SEED_SELECTION";
        case RuntimePhase::Battle: return "BATTLE";
        case RuntimePhase::Lost: return "LOST";
        case RuntimePhase::Award: return "AWARD";
        case RuntimePhase::Credit: return "CREDIT";
        case RuntimePhase::Challenge: return "CHALLENGE";
        case RuntimePhase::LevelComplete: return "LEVEL_COMPLETE";
        default: return "UNKNOWN";
    }
}

const char* RuntimePhaseReader::raw_scene_name(std::int32_t raw_scene) {
    switch (raw_scene) {
        case 0: return "SCENE_LOADING";
        case 1: return "SCENE_MENU";
        case 2: return "SCENE_LEVEL_INTRO";
        case 3: return "SCENE_PLAYING";
        case 4: return "SCENE_ZOMBIES_WON";
        case 5: return "SCENE_AWARD";
        case 6: return "SCENE_CREDIT";
        case 7: return "SCENE_CHALLENGE";
        default: return "SCENE_UNKNOWN";
    }
}

RuntimeState RuntimePhaseReader::read() {
    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();
    if (!pvz.GameOn() && !pvz.FindPvZ()) {
        return RuntimeState{};
    }
    auto state = read_nolock(pvz);
    mgr.sync_cheat_keys_for_scene_nolock(state.raw_scene);
    return state;
}

RuntimeState RuntimePhaseReader::read_nolock(PvzRuntime::VersionedPvZ& pvz) {
    RuntimeState state;
    if (!pvz.GameOn()) return state;
    pvz.ensure_remote_idle();

    state.process_available = true;
    state.raw_scene = pvz.read_memory_checked<int>({pvz.data().lawn, pvz.data().game_ui});

    const std::uintptr_t lawn = pvz.read_memory_checked<std::uintptr_t>({pvz.data().lawn});
    if (lawn) {
        state.game_mode = pvz.read_memory_checked<int>({lawn + pvz.data().game_mode});
        if (pvz.version_id() == PVZ_1_2_0_1065_EN) state.dave_message = pvz.read_memory_checked<int>({lawn + 0x850});

        // Read the authoritative Adventure-loop counter from PlayerInfo instead
        // of guessing first/second playthrough from whether a SeedChooser exists.
        const std::uintptr_t player_info =
            pvz.read_memory_checked<std::uintptr_t>({lawn + kLawnPlayerInfoOffset});
        if (player_info) {
            state.finished_adventure =
                pvz.read_memory_checked<int>({player_info + kPlayerFinishedAdventureOffset});
        }

        const std::uintptr_t board = pvz.read_memory_checked<std::uintptr_t>({lawn + pvz.data().board});
        state.board_present = board != 0;
        state.board_address = board;

        if (board) {
            state.level_id = pvz.read_memory_checked<int>({board + pvz.data().adventure_level});
            state.game_clock = pvz.read_memory_checked<int>({board + pvz.data().game_clock});
            state.game_clock_available = true;

            // This Board region follows the same version delta convention used
            // by reader_global.cpp. 1.2.0.1065 EN currently has delta 0.
            const std::uintptr_t board_var_offset = pvz.data().sun - 0x5560;
            state.level_complete = pvz.read_memory_checked<bool>({
                board + kBoardLevelCompleteOffset + board_var_offset});

            const std::uintptr_t cutscene =
                pvz.read_memory_checked<std::uintptr_t>({board + pvz.data().cut_scene});
            state.cutscene_present = cutscene != 0;

            if (cutscene) {
                state.seed_choosing =
                    pvz.read_memory_checked<bool>({cutscene + kCutsceneSeedChoosingOffset});

                const int cutscene_time =
                    pvz.read_memory_checked<int>({cutscene + kCutsceneTimeOffset});
                state.cutscene_time = cutscene_time;
                const int crazy_dave_time =
                    pvz.read_memory_checked<int>({cutscene + kCutsceneCrazyDaveTimeOffset});

                state.dave_talking =
                    state.raw_scene == 2 && crazy_dave_time > 0 &&
                    cutscene_time < (kTimePanRightEnd + crazy_dave_time);
            }
        }
    }

    state.phase = classify(
        state.raw_scene,
        state.seed_choosing,
        state.dave_talking,
        state.level_complete,
        state.game_mode);
    state.status = to_game_status(state.phase);
    return state;
}

}  // namespace pvz::game
