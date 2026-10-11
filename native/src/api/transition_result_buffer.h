#pragma once

#include "serializer.h"
#include <atomic>
#include <cstdint>
#include <string>
#include <utility>

namespace pvz::api {

// One pending result per calling thread. A new request invalidates its predecessor.
// Request IDs are unique across threads; a getter never executes a transition.
class TransitionResultBuffer {
public:
    std::uint64_t begin() {
        clear();
        active_id_ = ++next_id_;
        if (!active_id_) active_id_ = ++next_id_;
        return active_id_;
    }

    int publish(std::string payload, char* out, int capacity) {
        pending_id_ = active_id_;
        active_id_ = 0;
        payload_ = std::move(payload);
        const int result = Serializer::copy_to_c_buffer(payload_, out, capacity);
        // Legacy callers do not begin a request; never expose their result under an old ID.
        if (result >= -1 || !pending_id_) clear();
        return result;
    }

    int read(std::uint64_t id, char* out, int capacity) {
        if (!id || id != pending_id_) return -1;
        const int result = Serializer::copy_to_c_buffer(payload_, out, capacity);
        if (result >= 0) clear();
        return result;
    }

private:
    void clear() { active_id_ = pending_id_ = 0; payload_.clear(); }
    inline static std::atomic<std::uint64_t> next_id_{0};
    std::uint64_t active_id_ = 0, pending_id_ = 0;
    std::string payload_;
};

inline thread_local TransitionResultBuffer transition_results;

}  // namespace pvz::api
