#pragma once
#include "pvz/runtime.h"
namespace pvz::runtime {
class NativeRuntime final : public ExecutionBackend {
public:
    static NativeRuntime& instance();
    RuntimeCapabilities capabilities() const noexcept override;
    Status configure(const RuntimeOptions& options) override;
    Status reset(const ResetConfig&, ResetResult&) override;
    Status advance_phase(const PhaseAdvanceRequest&, PhaseAdvanceResult&) override;
    Status read_snapshot(RuntimeSnapshot&) override;
    bool read_cached_snapshot(RuntimeSnapshot&);
    Status step(const Action&, int, StepResult&) override;
    const RuntimeOptions& options() const noexcept { return options_; }
    bool prepare();
    bool begin_lifecycle();
    const std::string& lifecycle_error() const noexcept { return lifecycle_error_; }
    bool finish_lifecycle();
    bool service();
    Status close();
    void new_episode() noexcept;
    void invalidate() noexcept { ++revision_; cached_ = false; }
    void cache_transition(const Observation&, const std::vector<bool>&);
private:
    RuntimeOptions options_;
    std::uint64_t episode_ = 0, revision_ = 0;
    RuntimeSnapshot snapshot_;
    std::uint64_t snapshot_commands_ = 0;
    bool cached_ = false;
    std::string lifecycle_error_;
};
} // namespace pvz::runtime
