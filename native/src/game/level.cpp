#include "runtime/native_errors.h"
#include "pvz/api.h"
#include "../process/runtime.h"

extern "C" bool PVZ_GYM_CALL pvz_gym_attach() { return pvz::process::attach(); }
extern "C" bool PVZ_GYM_CALL pvz_gym_is_game_on() { return pvz::process::game_on(); }
extern "C" int PVZ_GYM_CALL pvz_gym_get_ui() { return pvz::process::ui(); }

extern "C" int PVZ_GYM_CALL pvz_gym_set_adventure_playthrough(int playthrough) {
    return pvz::process::set_adventure_playthrough(playthrough);
}

extern "C" int PVZ_GYM_CALL pvz_gym_select_adventure_level(int area, int sub_level) {
    if (area < 1 || area > 5 || sub_level < 1 || sub_level > 10) return 1002;
    try { return pvz::process::select_level((area - 1) * 10 + sub_level); }
    catch (const PvzRuntime::NativeOperationError& e) { return static_cast<int>(e.error()); }
    catch (...) { return static_cast<int>(pvz::ActionError::Unsupported); }
}

extern "C" int PVZ_GYM_CALL pvz_gym_select_adventure_level_id(int level_id) {
    try { return pvz::process::select_level(level_id); }
    catch (const PvzRuntime::NativeOperationError& e) { return static_cast<int>(e.error()); }
    catch (...) { return static_cast<int>(pvz::ActionError::Unsupported); }
}

extern "C" int PVZ_GYM_CALL pvz_gym_get_current_adventure_level(int* area, int* sub_level, int* level_id, int* ui) {
    if (area) *area = 0;
    if (sub_level) *sub_level = 0;
    if (level_id) *level_id = 0;
    if (ui) *ui = pvz::process::ui();
    if (!pvz::process::game_on()) return 1001;
    const int current = pvz::process::current_level();
    if (level_id) *level_id = current;
    if (area) *area = (current - 1) / 10 + 1;
    if (sub_level) *sub_level = (current - 1) % 10 + 1;
    return 0;
}

extern "C" const char* PVZ_GYM_CALL pvz_gym_error_message(int code) {
    switch (code) {
        case 0:    return "success";
        // 进程/关卡错误
        case 1001: return "PlantsVsZombies.exe was not found";
        case 1002: return "invalid argument";
        // ActionError 全码覆盖
        case 1003: return "insufficient sun";
        case 1004: return "cooldown active";
        case 1005: return "slot index out of range";
        case 1006: return "invalid grid coordinates";
        case 1007: return "terrain requires lily pad";
        case 1008: return "terrain requires flower pot";
        case 1009: return "grid blocked by hazard";
        case 1010: return "main plant already exists";
        case 1011: return "upgrade base mismatch";
        case 1012: return "cob cannon double base invalid";
        case 1013: return "pumpkin already equipped";
        case 1014: return "no plant to shovel";
        case 1015: return "cob cannon recharging";
        case 1016: return "cob target out of arena";
        case 1017: return "game not in battle state";
        case 1018: return "cob cannon not found";
        case 1019: return "action timeout";
        case 1020: return "unsupported game version";
        case 1021: return "card unavailable";
        case 1099: return "unsupported action";
        default:   return "unknown native PvZ Gym error";
    }
}
