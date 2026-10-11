#pragma once
#include "pvz/types.h"
#include <cstring>
#include <type_traits>

namespace pvz::api {
// PVZB v1: little-endian typed values, with length-prefixed maps/arrays/UTF-8.
// This writes native facts directly; no JSON is produced on the hot path.
class BinaryWriter {
public:
    std::string bytes = "PVZB\1";
    template<class T> void raw(T value) { bytes.append(reinterpret_cast<const char*>(&value), sizeof(value)); }
    void value(bool v) { raw<std::uint8_t>(v ? 2 : 1); }
    template<class T, std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>, int> = 0>
    void value(T v) { raw<std::uint8_t>(3); raw<std::int64_t>(static_cast<std::int64_t>(v)); }
    template<class T, std::enable_if_t<std::is_enum_v<T>, int> = 0>
    void value(T v) { value(static_cast<std::int32_t>(v)); }
    void value(float v) { raw<std::uint8_t>(4); raw<double>(v); }
    void value(const std::string& v) { raw<std::uint8_t>(5); raw<std::uint32_t>(static_cast<std::uint32_t>(v.size())); bytes += v; }
    void value(const char* v) { value(std::string(v)); }
    template<class T> void value(const std::vector<T>& v) {
        raw<std::uint8_t>(6); raw<std::uint32_t>(static_cast<std::uint32_t>(v.size()));
        for (const auto& item : v) value(item);
    }
    void value(const std::vector<bool>& v) {
        raw<std::uint8_t>(6); raw<std::uint32_t>(static_cast<std::uint32_t>(v.size()));
        for (bool item : v) value(item);
    }
    template<class T> void field(const char* key, const T& item, std::uint32_t& count) { value(key); value(item); ++count; }
    std::size_t begin_map() { raw<std::uint8_t>(7); const auto at = bytes.size(); raw<std::uint32_t>(0); return at; }
    void end_map(std::size_t at, std::uint32_t count) { std::memcpy(bytes.data() + at, &count, 4); }
    void null() { raw<std::uint8_t>(0); }
    void value(const GlobalState& v);
    void value(const CardState& v);
    void value(const PlantState& v);
    void value(const ZombieState& v);
    void value(const BossProjectileState& v);
    void value(const DroppedCardState& v);
    void value(const GridItemState& v);
    void value(const IceTrailState& v);
    void value(const MowerState& v);
    void value(const GridSquareState& v);
    void value(const Observation& v);
    void value(const StepEvents& v);
    void value(const ActionResult& v);
    void value(const ResetResult& v);
    void value(const StepResult& v);
};
template<class T> std::string binary_result(const T& value) { BinaryWriter writer; writer.value(value); return std::move(writer.bytes); }
} // namespace pvz::api
