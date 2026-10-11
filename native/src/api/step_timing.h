#pragma once
#include <atomic>
#include <array>
#include <chrono>
#include <sstream>
#include <locale>
#include <string>

namespace pvz::api {
inline std::atomic<bool> step_profiling{false};
inline thread_local std::string last_step_timing = "{}";
class StepTiming {
    bool enabled_ = step_profiling.load(std::memory_order_relaxed);
    std::array<double, 7> ms_{};
public:
    enum Phase { Before, Action, Automation, Tick, After, Mask, Serialize };
    ~StepTiming() {
        if (!enabled_) return;
        const char* names[] = {"before_observation", "action", "automation", "tick", "after_observation", "mask", "serialization"};
        std::ostringstream out; out.imbue(std::locale::classic()); out << '{';
        for (int i = 0; i < 7; ++i) { if (i) out << ','; out << '"' << names[i] << "\":" << ms_[i]; }
        out << '}'; last_step_timing = out.str();
    }
    template<class F> auto measure(Phase phase, F call) -> decltype(call()) {
        if (!enabled_) return call();
        const auto start = std::chrono::steady_clock::now();
        struct End {
            double& ms; std::chrono::steady_clock::time_point start;
            ~End() { ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(); }
        } end{ms_[phase], start};
        return call();
    }
};
inline thread_local StepTiming* current_step_timing = nullptr;
struct StepTimingScope {
    StepTiming* previous;
    explicit StepTimingScope(StepTiming& timing) : previous(current_step_timing) { current_step_timing = &timing; }
    ~StepTimingScope() { current_step_timing = previous; }
};
template<class F> auto measure_step(StepTiming::Phase phase, F call) -> decltype(call()) {
    return current_step_timing ? current_step_timing->measure(phase, call) : call();
}
}
