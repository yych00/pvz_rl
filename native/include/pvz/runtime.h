#pragma once

#include "pvz/types.h"

#include <cstdint>

namespace pvz::game {

// Public phase IDs match RuntimePhaseReader; raw GameUI=2 contains three phases.
enum class RuntimePhase : std::int32_t {
    Unknown = 0, Offline = 1, Loading = 2, Menu = 3,
    LevelIntro = 4, Dialogue = 5, SeedSelection = 6, Battle = 7,
    Lost = 8, Award = 9, Credit = 10, Challenge = 11, LevelComplete = 12,
};

}  // namespace pvz::game

namespace pvz::runtime {

inline constexpr int kInterfaceVersion = 1;

enum class Status : int { Ok = 0, Unsupported = 1, InvalidArgument = 2, InvalidState = 3 };
enum class ExecutorKind : int { Legacy = 0, Resident = 1 };
enum class TransportKind : int { Json = 0, Binary = 1, SharedMemory = 2 };
enum class SnapshotKind : int { Legacy = 0, Shared = 1 };

struct RuntimeOptions {
    // 0 retains current automatic attachment; a positive PID requires capability support.
    std::uint32_t process_id = 0;
    ExecutorKind executor = ExecutorKind::Legacy;
    TransportKind transport = TransportKind::Json;
    SnapshotKind snapshot = SnapshotKind::Legacy;
};

struct RuntimeCapabilities {
    int interface_version = kInterfaceVersion;
    bool stepped = false;
    bool headless = false;
    bool continuous = false;
    bool attach_pid = false;
    bool resident_executor = false;
    bool binary_transport = false;
    bool shared_memory_transport = false;
    bool shared_snapshot = false;
    bool turbo = false;
};

enum TurboFlag : std::uint32_t {
    SuppressRender = 1u << 0,
    ServiceMessages = 1u << 1,
    FastIntro = 1u << 2,
    FastTerminal = 1u << 3,
    MuteAudio = 1u << 4,
};
inline constexpr std::uint32_t kTurboFlags =
    SuppressRender | ServiceMessages | FastIntro | FastTerminal | MuteAudio;

struct TurboOptions {
    std::uint32_t flags = kTurboFlags;
};

struct PhaseAdvanceRequest {
    game::RuntimePhase expected_phase = game::RuntimePhase::Unknown;
    int updates = 0;
};

struct PhaseAdvanceResult {
    game::RuntimePhase phase_before = game::RuntimePhase::Unknown;
    game::RuntimePhase phase_after = game::RuntimePhase::Unknown;
    int updates_executed = 0;
    // -1 means no battle-clock measurement. Intro updates are not battle ticks.
    std::int64_t game_ticks_advanced = -1;
};

struct SnapshotIdentity {
    std::uint32_t process_id = 0;
    std::uint64_t episode_id = 0;
    std::uint64_t board_id = 0;
    std::uint64_t revision = 0;
};

struct RuntimeSnapshot {
    SnapshotIdentity identity;
    Observation observation;
    std::vector<bool> action_mask;
};

// Mechanisms are shared across policies. No implementation is implied by these
// interfaces: each backend must advertise what it actually supports.
class ExecutionBackend {
public:
    virtual ~ExecutionBackend() = default;
    virtual RuntimeCapabilities capabilities() const noexcept = 0;
    virtual Status configure(const RuntimeOptions& options) = 0;
    virtual Status reset(const ResetConfig& config, ResetResult& result) = 0;
    virtual Status advance_phase(const PhaseAdvanceRequest& request, PhaseAdvanceResult& result) = 0;
    virtual Status read_snapshot(RuntimeSnapshot& snapshot) = 0;
    virtual Status step(const Action& action, int ticks, StepResult& result) = 0;
};

class ModePolicy {
public:
    virtual ~ModePolicy() = default;
    // Enter before reset to cover all lifecycle phases; leave must be idempotent.
    virtual Status enter(ExecutionBackend& backend) = 0;
    virtual Status leave() = 0;
    virtual Status advance_phase(const PhaseAdvanceRequest& request, PhaseAdvanceResult& result) = 0;
    virtual Status step(const Action& action, int ticks, StepResult& result) = 0;
};

}  // namespace pvz::runtime
