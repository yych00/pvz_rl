#pragma once

#include "board_update_batch.h"
#include <Windows.h>

namespace pvz::game {

// Gym-owned injector: leaves the upstream Pt::Code implementation untouched.
int execute_board_update_batch(
    HANDLE process, const BoardUpdateBatchCode& stub, std::uint32_t max_updates);

} // namespace pvz::game
