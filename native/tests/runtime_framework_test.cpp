#include "pvz/api.h"
#include "test_check.h"
#include <string>
#include <vector>
int main() {
    char tiny[2];
    int required = -pvz_gym_get_runtime_capabilities_json(tiny, 2);
    CHECK(required > 2);
    std::vector<char> buffer(required);
    CHECK(pvz_gym_get_runtime_capabilities_json(buffer.data(), required) == required-1);
    CHECK(std::string(buffer.data()).find("\"turbo\"") != std::string::npos);
    CHECK(pvz_gym_configure_runtime(0, -1, 0, 0) == 2);
    CHECK(pvz_gym_begin_turbo_mode(1u << 31) == 2);
    CHECK(pvz_gym_begin_turbo_mode(0) == 2);
    CHECK(pvz_gym_begin_turbo_mode(1) == 2);
    CHECK(pvz_gym_begin_turbo_mode(2) == 2);
    CHECK(pvz_gym_end_turbo_mode() == 0);
    // Bind an impossible PID before all game-access checks. Never touch an
    // interactive game that happens to be running on the testing machine.
    CHECK(pvz_gym_configure_runtime(0xffffffffu, 0, 0, 0) == 0);
    CHECK(!pvz_gym_attach());
    CHECK(pvz_gym_begin_turbo_mode(31) == 3);
    required = -pvz_gym_get_runtime_error_json(tiny, 2);
    CHECK(required > 2);
    buffer.resize(required);
    CHECK(pvz_gym_get_runtime_error_json(buffer.data(), required) == required-1);
    const std::string failure(buffer.data());
    CHECK(failure.find("target game process is unavailable") != std::string::npos);
    CHECK(failure.find("\"operation\":\"begin_turbo_mode\"") != std::string::npos);
    CHECK(failure.find("\"executor\":{") != std::string::npos);
    // Read-only diagnostic queries and lifecycle cleanup retain the frozen error.
    CHECK(pvz_gym_end_turbo_mode() == 0);
    CHECK(pvz_gym_get_runtime_error_json(buffer.data(), required) == required-1);
    CHECK(std::string(buffer.data()) == failure);
    auto request = pvz_gym_begin_result_request();
    required = -pvz_gym_env_step_turbo(0, -1, -1, -1, 0, -1, -1, 150, tiny, 2);
    CHECK(required > 2);
    buffer.resize(required);
    CHECK(pvz_gym_read_result_request(request, buffer.data(), required) == required-1);
    CHECK(std::string(buffer.data()).find("Turbo is not active") != std::string::npos);
    CHECK(pvz_gym_read_result_request(request, buffer.data(), required) == -1);
    char empty_error[3];
    CHECK(pvz_gym_get_runtime_error_json(empty_error, 3) == 2);
    CHECK(std::string(empty_error) == "{}");
    CHECK(pvz_gym_configure_runtime(0xffffffffu, 1, 2, 1) == 0);
    CHECK(pvz_gym_end_turbo_mode() == 0);
}
