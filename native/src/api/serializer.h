#pragma once

#include "pvz/types.h"

#include <string>

namespace pvz::api {

class Serializer {
public:
    static std::string observation_to_json(const Observation& obs);
    static std::string reset_result_to_json(const ResetResult& res);
    static std::string step_result_to_json(const StepResult& res);
    static int copy_to_c_buffer(const std::string& src, char* out_buf, int max_len);
};

}  // namespace pvz::api
