#include "reader_internal.h"
#include "plant_snapshot.h"
namespace pvz::observation {
bool read_plants_nolock(PvzRuntime::VersionedPvZ& pvz, uintptr_t board, int rows, std::vector<PlantState>& out) {
    out.clear();
    if (!board || pvz.data().plant_dead != 0x141 || pvz.data().plant_squished != 0x142 ||
        pvz.data().plant_asleep != 0x143 || pvz.data().plant_row != 0x1c ||
        pvz.data().plant_col != 0x28 || pvz.data().plant_type != 0x24) return false;
    return read_plant_snapshot([&](uintptr_t address, void* bytes, std::size_t size) {
        return pvz.read_remote_bytes(address, bytes, size);
    }, board + pvz.data().plant, board + pvz.data().plant_count_max, rows, out);
}
}
