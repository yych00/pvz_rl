#pragma once
#include "binary_serializer.h"
#include "serializer.h"
#include "shared_result.h"
#include "../runtime/native_runtime.h"
namespace pvz::api {
inline std::string wire_result(const StepResult& result) {
    const auto transport = runtime::NativeRuntime::instance().options().transport;
    if (transport == runtime::TransportKind::Json) return Serializer::step_result_to_json(result);
    const auto bytes = binary_result(result);
    return transport == runtime::TransportKind::SharedMemory ? shared_result(bytes) : bytes;
}
inline std::string wire_result(const ResetResult& result) {
    const auto transport = runtime::NativeRuntime::instance().options().transport;
    if (transport == runtime::TransportKind::Json) return Serializer::reset_result_to_json(result);
    const auto bytes = binary_result(result);
    return transport == runtime::TransportKind::SharedMemory ? shared_result(bytes) : bytes;
}
} // namespace pvz::api
