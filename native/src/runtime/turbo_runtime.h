#pragma once

#include "pvz/runtime.h"
#include <string>

namespace pvz::runtime {

class TurboRuntime final : public ModePolicy {
public:
    explicit TurboRuntime(TurboOptions options = {}) : options_(options) {}
    static TurboRuntime& instance();
    Status set_options(TurboOptions options) {
        if (options.flags & ~kTurboFlags || !(options.flags & SuppressRender) ||
            !(options.flags & ServiceMessages)) return Status::InvalidArgument;
        if (active() && options.flags != options_.flags) return Status::InvalidState;
        options_ = options;
        return Status::Ok;
    }
    bool active() const noexcept { return backend_ != nullptr; }
    const std::string& enter_error() const noexcept { return enter_error_; }
    bool fast_intro() const noexcept { return options_.flags & FastIntro; }
    bool fast_terminal() const noexcept { return options_.flags & FastTerminal; }
    // Called between lifecycle operations; never while an observation is read.
    bool service();
    Status enter(ExecutionBackend& backend) override;
    Status leave() override;
    Status advance_phase(const PhaseAdvanceRequest& request, PhaseAdvanceResult& result) override;
    Status step(const Action& action, int ticks, StepResult& result) override;

private:
    TurboOptions options_;
    ExecutionBackend* backend_ = nullptr;
    std::uint32_t original_board_ = 0;
    bool original_pause_ = false;
    std::string enter_error_;
};

}  // namespace pvz::runtime
