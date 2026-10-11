#include "seed_selection_executor.h"

#include "seed_selection_reader.h"
#include "seed_selection_validator.h"

#include "../game/simulation_clock.h"
#include "../process/process_manager.h"
#include "../automation/automation_controller.h"
#include "../automation/intro_skipper.h"

#include <mutex>

namespace pvz::seed_selection {
namespace {

constexpr uintptr_t kCutsceneSeedChoosingOffset = 0x2C;

SeedSelectionResult make_error(SeedSelectionError error, const char* message) {
    return SeedSelectionResult{false, error, message};
}

bool is_seed_choosing_nolock(PvzRuntime::VersionedPvZ& pvz, uintptr_t lawn) {
    if (!pvz.GameOn() || !lawn || pvz.GameUI() != 2) return false;
    const uintptr_t board = pvz.ReadMemory<uintptr_t>({lawn + pvz.data().board});
    if (!board) return false;
    const uintptr_t cutscene = pvz.ReadMemory<uintptr_t>({board + pvz.data().cut_scene});
    if (!cutscene) return false;
    return pvz.ReadMemory<bool>({cutscene + kCutsceneSeedChoosingOffset});
}

uintptr_t get_chooser_and_seed(int plant_type, uintptr_t& out_seed_addr) {
    out_seed_addr = 0;
    if (plant_type < 0 || plant_type >= kNumSeedsInChooser) {
        return 0;
    }
    auto& mgr = process::ProcessManager::instance();
    auto& pvz = mgr.runtime();

    const uintptr_t lawn = pvz.ReadMemory<uintptr_t>({pvz.data().lawn});
    if (!is_seed_choosing_nolock(pvz, lawn)) return 0;

    const uintptr_t chooser = pvz.ReadMemory<uintptr_t>({lawn + 0x774});
    if (!chooser) return 0;

    const uintptr_t addr = chooser + 0xa4 + static_cast<uintptr_t>(plant_type * 0x3c);
    const int internal_type = pvz.ReadMemory<int>({addr + 0x20});
    if (internal_type >= 0 && internal_type < kSeedTypeCapacity && internal_type != plant_type) {
        // Sanity warning: internal seed type differs from chooser slot index.
    }
    out_seed_addr = addr;
    return chooser;
}

// 瞬间将所有空中飞行的卡片（0:飞向卡槽, 2:飞回选择盘）安全落地并清除飞行计数，根除格子冲突与重叠
void land_all_flying_seeds(PvzRuntime::VersionedPvZ& pvz, uintptr_t chooser) {
    if (!chooser || !pvz.GameOn() || pvz.GameUI() != 2) return;

    bool found_flying = false;
    for (int i = 0; i < kNumSeedsInChooser; ++i) {
        uintptr_t addr = chooser + 0xa4 + static_cast<uintptr_t>(i * 0x3c);
        int st = pvz.ReadMemory<int>({addr + 0x24});
        if (st == 0 || st == 2) {
            found_flying = true;
            // 调用原版 LandFlyingSeed (0x485320): EAX = seed_addr, ECX = chooser
            pvz.asm_init();
            pvz.asm_add_byte(0x60); // pushad
            pvz.asm_mov_exx(Pt::Reg::EAX, static_cast<unsigned int>(addr));
            pvz.asm_mov_exx(Pt::Reg::ECX, static_cast<unsigned int>(chooser));
            pvz.asm_call(0x485320);
            pvz.asm_add_byte(0x61); // popad
            pvz.asm_ret();
            pvz.asm_code_inject();
        }
    }

    const int flight_cnt = pvz.ReadMemory<int>({chooser + 0xd20});
    if (found_flying || flight_cnt != 0) {
        pvz.WriteMemory<int>(0, {chooser + 0xd20});
        // 刷新开始按钮 (0x486a30 自带 ret 4，由被调方清栈，调用方切勿 add esp, 4)
        pvz.asm_init();
        pvz.asm_add_byte(0x60); // pushad
        pvz.asm_push_dword(static_cast<unsigned int>(chooser));
        pvz.asm_call(0x486a30);
        pvz.asm_add_byte(0x61); // popad
        pvz.asm_ret();
        pvz.asm_code_inject();
    }
}

}  // namespace

static SeedSelectionResult select_seed_impl(int plant_type, int imitater_target) {
    SeedSelectionState state;
    if (!SeedSelectionReader::read(state)) {
        return make_error(SeedSelectionError::GameNotFound, "game is not attached");
    }
    const auto error = imitater_target >= 0
        ? SeedSelectionValidator::can_select_imitater(imitater_target, state)
        : SeedSelectionValidator::can_select(plant_type, state);
    if (error != SeedSelectionError::None) {
        return make_error(error, "seed selection action is not currently legal");
    }

    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();

    uintptr_t seed_addr = 0;
    const uintptr_t chooser = get_chooser_and_seed(plant_type, seed_addr);
    if (!chooser || !seed_addr) {
        return make_error(SeedSelectionError::SeedNotAvailable, "seed packet address not found or chooser no longer interactive");
    }

    land_all_flying_seeds(pvz, chooser);

    pvz.asm_init();
    pvz.asm_add_byte(0x60); // pushad

    pvz.asm_push_dword(static_cast<unsigned int>(seed_addr));
    pvz.asm_mov_exx(Pt::Reg::EAX, static_cast<unsigned int>(chooser));
    pvz.asm_call(0x486380);

    pvz.asm_mov_exx(Pt::Reg::EAX, static_cast<unsigned int>(seed_addr));
    pvz.asm_mov_exx(Pt::Reg::ECX, static_cast<unsigned int>(chooser));
    pvz.asm_call(0x485320);

    pvz.asm_push_dword(static_cast<unsigned int>(chooser));
    pvz.asm_call(0x486a30);

    pvz.asm_add_byte(0x61); // popad
    pvz.asm_ret();
    pvz.asm_code_inject();

    land_all_flying_seeds(pvz, chooser);
    pvz.WriteMemory<int>(0, {chooser + 0xd20});

    if (imitater_target >= 0) {
        // ChosenSeed::mImitaterType. The original chooser can default this to
        // the last selected card, so override it with the encoded target.
        pvz.WriteMemory<int>(imitater_target, {seed_addr + 0x34});
        if (pvz.ReadMemory<int>({seed_addr + 0x34}) != imitater_target) {
            return make_error(SeedSelectionError::Unsupported, "imitater target was not applied");
        }
    }

    return SeedSelectionResult{true, SeedSelectionError::None, "seed selected successfully"};
}

SeedSelectionResult SeedSelectionExecutor::select_seed(int plant_type) {
    return select_seed_impl(plant_type, -1);
}

SeedSelectionResult SeedSelectionExecutor::select_imitater(int target_type) {
    return select_seed_impl(kImitaterType, target_type);
}

SeedSelectionResult SeedSelectionExecutor::deselect_seed(int plant_type) {
    SeedSelectionState state;
    if (!SeedSelectionReader::read(state)) {
        return make_error(SeedSelectionError::GameNotFound, "game is not attached");
    }
    const auto error = SeedSelectionValidator::can_deselect(plant_type, state);
    if (error != SeedSelectionError::None) {
        return make_error(error, "seed deselection action is not currently legal");
    }

    auto& mgr = process::ProcessManager::instance();
    std::lock_guard<std::mutex> lock(mgr.mutex());
    auto& pvz = mgr.runtime();

    uintptr_t seed_addr = 0;
    const uintptr_t chooser = get_chooser_and_seed(plant_type, seed_addr);
    if (!chooser || !seed_addr) {
        return make_error(SeedSelectionError::SeedNotSelected, "seed packet address not found or chooser no longer interactive");
    }

    land_all_flying_seeds(pvz, chooser);

    pvz.asm_init();
    pvz.asm_add_byte(0x60); // pushad

    pvz.asm_push_dword(static_cast<unsigned int>(seed_addr));
    pvz.asm_mov_exx(Pt::Reg::EAX, static_cast<unsigned int>(chooser));
    pvz.asm_call(0x4861e0);

    pvz.asm_mov_exx(Pt::Reg::EAX, static_cast<unsigned int>(seed_addr));
    pvz.asm_mov_exx(Pt::Reg::ECX, static_cast<unsigned int>(chooser));
    pvz.asm_call(0x485320);

    pvz.asm_push_dword(static_cast<unsigned int>(chooser));
    pvz.asm_call(0x486a30);

    pvz.asm_add_byte(0x61); // popad
    pvz.asm_ret();
    pvz.asm_code_inject();

    land_all_flying_seeds(pvz, chooser);
    pvz.WriteMemory<int>(0, {chooser + 0xd20});

    return SeedSelectionResult{true, SeedSelectionError::None, "seed deselected successfully"};
}

SeedSelectionResult SeedSelectionExecutor::confirm_seeds() {
    SeedSelectionState state;
    if (!SeedSelectionReader::read(state)) {
        return make_error(SeedSelectionError::GameNotFound, "game is not attached");
    }
    const auto error = SeedSelectionValidator::can_confirm(state);
    if (error != SeedSelectionError::None) {
        return make_error(error, "seed selection is not ready to confirm");
    }

    {
        auto& mgr = process::ProcessManager::instance();
        std::lock_guard<std::mutex> lock(mgr.mutex());
        auto& pvz = mgr.runtime();

        const uintptr_t lawn = pvz.ReadMemory<uintptr_t>({pvz.data().lawn});
        if (!lawn) return make_error(SeedSelectionError::GameNotFound, "lawnapp not found");
        if (!is_seed_choosing_nolock(pvz, lawn)) {
            return make_error(SeedSelectionError::InvalidPhase, "seed chooser is no longer interactive");
        }
        const uintptr_t chooser = pvz.ReadMemory<uintptr_t>({lawn + 0x774});
        if (!chooser) return make_error(SeedSelectionError::InvalidPhase, "chooser not found");

        land_all_flying_seeds(pvz, chooser);

        // SeedChooserScreen::CloseSeedChooser writes the bank and calls
        // CutScene::EndSeedChooser(), which clears mSeedChoosing.
        pvz.asm_init();
        pvz.asm_add_byte(0x60); // pushad
        pvz.asm_mov_exx(Pt::Reg::EBX, static_cast<unsigned int>(chooser));
        pvz.asm_call(0x487070);
        pvz.asm_add_byte(0x61); // popad
        pvz.asm_ret();
        pvz.asm_code_inject();
    }

    // 选卡关闭后立即尝试第二次 ESC；若界面尚未就绪，IntroSkipper::update()
    // 会在第二阶段按间隔重试。
    (void)automation::IntroSkipper::fast_forward_after_seed_selection();

    const auto cfg = automation::AutomationController::get_config();
    if (cfg.enabled && cfg.auto_dialogue) {
        automation::AutomationController::handle_prebattle();
    }

    return SeedSelectionResult{true, SeedSelectionError::None, "seed selection confirmed successfully"};
}

}  // namespace pvz::seed_selection
