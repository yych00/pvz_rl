#include "../src/api/transition_result_buffer.h"
#include "../src/api/step_timing.h"
#include "../src/observation/plant_snapshot.h"
#include "../src/process/barrier_ownership.h"
#include "test_check.h"
#include <thread>
#include <cstring>

template<class T> void put(std::vector<unsigned char>& b, std::size_t offset, T value) {
    std::memcpy(b.data() + offset, &value, sizeof(value));
}
DWORD WINAPI finish_thread(LPVOID) { return 0; }
int main() {
    // Exercise Windows handle rights and liveness, rather than a mocked flag.
    DWORD tid = 0;
    HANDLE created = CreateThread(nullptr, 0, finish_thread,
                                 nullptr, CREATE_SUSPENDED, &tid);
    CHECK(created != nullptr);
    HANDLE owned = OpenThread(THREAD_SUSPEND_RESUME | THREAD_QUERY_INFORMATION |
                              THREAD_GET_CONTEXT | SYNCHRONIZE, FALSE, tid);
    CHECK(owned != nullptr);
    CHECK(pvz::process::owns_live_thread(true, owned, GetCurrentProcessId()));
    CHECK(!pvz::process::owns_live_thread(false, owned, GetCurrentProcessId()));
    CHECK(!pvz::process::owns_live_thread(true, owned, GetCurrentProcessId() + 1));
    CHECK(!pvz::process::owns_live_thread(true, nullptr, GetCurrentProcessId()));
    CHECK(ResumeThread(owned) == 1);
    CHECK(WaitForSingleObject(owned, 5000) == WAIT_OBJECT_0);
    CHECK(!pvz::process::owns_live_thread(true, owned, GetCurrentProcessId()));
    CloseHandle(owned); CloseHandle(created);
    using namespace pvz::api;
    char tiny[2], large[1024];
    auto id = transition_results.begin();
    CHECK(transition_results.publish("first transition", tiny, 2) == -17);
    CHECK(transition_results.read(id + 1, large, sizeof(large)) == -1);
    std::thread other([&] { CHECK(transition_results.read(id, large, sizeof(large)) == -1); });
    other.join();
    CHECK(transition_results.read(id, tiny, 2) == -17);
    CHECK(transition_results.read(id, large, sizeof(large)) == 16);
    CHECK(std::string(large) == "first transition");
    CHECK(transition_results.read(id, large, sizeof(large)) == -1);
    id = transition_results.begin(); transition_results.publish("old", tiny, 2);
    const auto next = transition_results.begin();
    CHECK(next != id); CHECK(transition_results.read(id, large, sizeof(large)) == -1);
    CHECK(transition_results.publish("new", large, sizeof(large)) == 3);
    CHECK(transition_results.read(next, large, sizeof(large)) == -1);

    using namespace pvz::observation;
    std::vector<unsigned char> b(kPlantStride * 3);
    for (int i = 0; i < 3; ++i) {
        const auto off = i * kPlantStride;
        put(b, off + 0x148, std::uint32_t(0x81230000u | i));
        put(b, off + 0x24, i == 0 ? 16 : i == 1 ? 30 : 47);
        put(b, off + 0x1c, 2); put(b, off + 0x28, 4);
        put(b, off + 0x40, 100); put(b, off + 0x44, 300);
        put(b, off + 0x3c, 37); put(b, off + 0x138, 8);
        b[off + 0x143] = 1;
    }
    std::vector<pvz::PlantState> plants;
    decode_plants(b, 5, plants); CHECK(plants.size() == 3);
    CHECK(plants[0].id == static_cast<std::int32_t>(0x81230000u));
    CHECK(plants[0].layer == 0 && plants[1].layer == 2 && plants[2].layer == 1);
    CHECK(plants[2].state == 37 && plants[2].imitater_type == 8 && plants[2].is_sleeping);
    CHECK(plants[2].row == 2 && plants[2].col == 4 && plants[2].hp == 100 && plants[2].max_hp == 300);
    auto invalid = b;
    for (std::size_t offset : {std::size_t(0x141), std::size_t(0x142)}) {
        invalid = b; invalid[offset] = 1; decode_plants(invalid, 5, plants); CHECK(plants.size() == 2);
    }
    for (auto field : {0x148, 0x1c, 0x28, 0x24, 0x40}) {
        invalid = b; put(invalid, field, field == 0x148 ? 7 : field == 0x28 ? 9 : field == 0x1c ? 5 : field == 0x24 ? -1 : 0);
        decode_plants(invalid, 5, plants); CHECK(plants.size() == 2);
    }
    std::int32_t count = 3; std::uint32_t ptr = 0x1000; int reads = 0, fail = 0;
    auto read = [&](std::uintptr_t addr, void* out, std::size_t size) {
        ++reads; if (reads == fail) return false;
        const void* source = addr == 10 ? static_cast<void*>(&ptr) : addr == 20 ? static_cast<void*>(&count) : b.data();
        std::memcpy(out, source, size); return true;
    };
    CHECK(read_plant_snapshot(read, 10, 20, 5, plants) && plants.size() == 3 && reads == 3);
    for (fail = 1; fail <= 3; ++fail) { reads = 0; CHECK(!read_plant_snapshot(read, 10, 20, 5, plants)); CHECK(plants.empty()); }
    fail = 0;
    for (int bad : {-1, 1025}) { count = bad; reads = 0; CHECK(!read_plant_snapshot(read, 10, 20, 5, plants)); }
    count = 0; ptr = 0; reads = 0; CHECK(read_plant_snapshot(read, 10, 20, 5, plants) && reads == 2);
    count = 1; reads = 0; CHECK(!read_plant_snapshot(read, 10, 20, 5, plants));
    int calls = 0;
    { StepTiming timing; timing.measure(StepTiming::Action, [&] { ++calls; }); }
    CHECK(calls == 1 && last_step_timing == "{}");
    step_profiling = true;
    { StepTiming timing; timing.measure(StepTiming::Action, [&] { ++calls; }); }
    CHECK(calls == 2 && last_step_timing.find("serialization") != std::string::npos);
}
