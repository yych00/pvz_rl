#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pvz {

// 选卡面板中植物卡片固定槽位数 (原版 ChosenSeed 数组固定 49 项: 0..39 常规植物 + 40..47 升级植物 + 48 模仿者)
constexpr std::int32_t kNumSeedsInChooser = 49;

// 固定 ABI 容量，不代表原版实际解锁植物数量；真实可选项必须来自 SeedChooserScreen。
constexpr std::int32_t kSeedTypeCapacity = 64;
constexpr std::int32_t kImitaterType = 48;
constexpr std::int32_t kImitaterTargetCount = 40;
constexpr std::int32_t kSeedSelectBase = 0;
constexpr std::int32_t kSeedImitaterBase = kSeedTypeCapacity;
constexpr std::int32_t kSeedDeselectBase = 2 * kSeedTypeCapacity;
constexpr std::int32_t kSeedConfirmIndex = 3 * kSeedTypeCapacity;
constexpr std::int32_t kSeedActionCount = kSeedConfirmIndex + 1;

enum class SeedSelectionError : std::int32_t {
    None = 0,
    GameNotFound = 2001,
    InvalidArgument = 2002,
    InvalidPhase = 2003,
    SeedNotAvailable = 2004,
    SeedAlreadySelected = 2005,
    SeedSelectionFull = 2006,
    SeedNotSelected = 2007,
    SeedSelectionIncomplete = 2008,
    SeedsInFlight = 2009,
    Unsupported = 2099
};

enum class SeedSelectionActionType : std::int32_t {
    Select = 0,
    Deselect = 1,
    Confirm = 2,
    SelectImitater = 3
};

struct SeedOptionState {
    std::int32_t plant_type = -1;
    bool available = false;
    bool unlocked = false;
    bool selected = false;
    bool disabled = false;
    std::int32_t chooser_state = 0;
};

enum class SeedWarning : std::uint32_t {
    None               = 0,
    NoSunProducer      = 1 << 0,
    NoPoolSupport      = 1 << 1,
    NoFlowerPot        = 1 << 2,
    NoFlyerDefense     = 1 << 3,
    UpgradeBaseMissing = 1 << 4,
};

struct SeedSelectionState {
    bool present = false;
    std::int32_t required_count = 0;
    std::int32_t selected_count = 0;
    bool can_confirm = false;
    bool seeds_in_flight = false;
    std::uint32_t warning_flags = 0;
    std::int32_t imitater_target = -1;
    std::vector<std::int32_t> selected_seeds;
    std::vector<SeedOptionState> options;
    bool zombie_roster_available = false;
    std::vector<std::int32_t> zombie_types_in_level;
};

struct SeedSelectionResult {
    bool success = false;
    SeedSelectionError error = SeedSelectionError::Unsupported;
    std::string message;
};

}  // namespace pvz
