#include "runtime/checked_memory.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>
static void require(bool ok) { if (!ok) std::exit(1); }
int main() {
    std::map<uintptr_t, std::vector<unsigned char>> memory;
    auto store = [&](uintptr_t addr, auto value) { auto& bytes = memory[addr]; bytes.resize(sizeof(value)); std::memcpy(bytes.data(), &value, sizeof(value)); };
    auto read = [&](uintptr_t addr, void* dest, size_t size) { auto it=memory.find(addr); if(it==memory.end() || it->second.size()!=size) return false; std::memcpy(dest,it->second.data(),size); return true; };
    store(0x100, uint32_t(0x200)); store(0x204, int(0)); store(0x208, double(1.25));
    require(PvzRuntime::checked_memory_read<int>(read,{0x100,4})==0);
    require(PvzRuntime::checked_memory_read<uintptr_t>(read,{0x100})==0x200);
    require(PvzRuntime::checked_memory_read<double>(read,{0x100,8})==1.25);
    store(0x210,uint32_t(0)); require(PvzRuntime::checked_memory_read<uintptr_t>(read,{0x210})==0);
    for(auto offsets : {std::initializer_list<uintptr_t>{0x100,12}, {0x210,4}, {0x100,UINT32_MAX}}) {
        bool rejected=false; try { (void)PvzRuntime::checked_memory_read<int>(read, offsets); } catch(const PvzRuntime::NativeOperationError& e) { rejected=true; require(e.error()==pvz::ActionError::Unsupported); }
        require(rejected);
    }
    std::puts("checked reads: real zero, 32-bit pointers, 64-bit values, missing/null/overflow rejected");
}
