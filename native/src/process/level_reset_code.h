#pragma once

#include "code.h"
#include <cstdint>

namespace pvz::process {

// Called while the original window thread is parked outside UpdateAppStep.
// SafeDeleteWidget then records depth zero, so normal depth-one updates cannot
// reclaim its widgets. Run the original collector at this same safe depth.
inline void build_level_reset_code(Pt::Code& code, std::uint32_t lawn_address,
                                  std::uint32_t reset_fn, std::uint32_t collect_fn,
                                  std::uint32_t kill_dialog_fn = 0,
                                  std::uint32_t kill_menu_fn = 0,
                                  std::uint32_t kill_award_fn = 0) {
    code.asm_init();
    code.asm_push_exx(Pt::Reg::EBX);
    code.asm_push_exx(Pt::Reg::ESI);
    code.asm_push_exx(Pt::Reg::EDI);
    // Almanac is a modal Dialog, independent of Board. PreNewGame only
    // replaces Board, so close it through LawnApp before creating the episode.
    // Keep this inside the reset barrier for direct select_level callers too.
    if (kill_dialog_fn) {
        code.asm_push_byte(3); // DIALOG_ALMANAC
        code.asm_mov_exx_dword_ptr(Pt::Reg::ECX, lawn_address);
        code.asm_call(kill_dialog_fn); // __thiscall: ECX=app, ret 4.
    }
    auto collect = [&] {
        if (!collect_fn) return;
        code.asm_mov_exx_dword_ptr(Pt::Reg::EAX, lawn_address);
        code.asm_push_exx(Pt::Reg::EAX);
        code.asm_call(collect_fn); // Original collector: app on stack, ret 4.
    };
    collect(); // Reclaim any older remote-call widgets before allocating.
    // These original helpers remove widgets and enqueue SafeDeleteWidget;
    // never clear LawnApp pointers without removing their widgets first.
    for (auto fn : {kill_menu_fn, kill_award_fn}) {
        if (!fn) continue;
        code.asm_mov_exx_dword_ptr(Pt::Reg::ESI, lawn_address);
        code.asm_call(fn); // Verified 1.2.0.1065 EN: ESI=app, plain ret.
    }
    code.asm_push_byte(0);
    code.asm_push_byte(0);
    code.asm_mov_exx_dword_ptr(Pt::Reg::ESI, lawn_address);
    code.asm_call(reset_fn); // PreNewGame: ESI=app, ret 8.
    collect();
    code.asm_add_list(0x31, 0xC0); // xor eax,eax: successful thread exit.
    code.asm_pop_exx(Pt::Reg::EDI);
    code.asm_pop_exx(Pt::Reg::ESI);
    code.asm_pop_exx(Pt::Reg::EBX);
    code.asm_add_list(0xC2, 0x04, 0x00); // Thread procedure: ret 4.
}

} // namespace pvz::process
