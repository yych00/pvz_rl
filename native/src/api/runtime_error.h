#pragma once

#include <string>

namespace pvz::api {

// Frozen on the failing caller thread, before Python closes the worker/runtime.
// Reading or resizing diagnostics must never rerun a transition.
inline thread_local std::string runtime_error_json = "{}";

inline std::string json_error_string(const std::string& value) {
    std::string result = "\"";
    constexpr char hex[] = "0123456789abcdef";
    for (unsigned char ch : value) {
        if (ch == '"' || ch == '\\') { result += '\\'; result += ch; }
        else if (ch < 0x20) {
            result += "\\u00"; result += hex[ch >> 4]; result += hex[ch & 15];
        } else result += ch;
    }
    return result + "\"";
}

inline void clear_runtime_error() { runtime_error_json = "{}"; }

inline void record_runtime_error(const char* operation, const char* kind, int code,
    const std::string& message, const std::string& executor) {
    runtime_error_json = "{\"operation\":" + json_error_string(operation) +
        ",\"kind\":" + json_error_string(kind) + ",\"code\":" + std::to_string(code) +
        ",\"message\":" + json_error_string(message) + ",\"executor\":" + executor + "}";
}

} // namespace pvz::api
