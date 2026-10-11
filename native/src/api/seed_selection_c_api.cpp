#include "runtime/native_errors.h"
#include "pvz/api.h"
#include "pvz/seed_selection.h"

#include "serializer.h"
#include "../seed_selection/seed_selection_executor.h"
#include "../seed_selection/seed_selection_reader.h"
#include "../seed_selection/seed_selection_serializer.h"
#include "../seed_selection/seed_selection_validator.h"

namespace {

int write_result_message(const pvz::SeedSelectionResult& result, char* out_msg_buf, int max_msg_len) {
    if (out_msg_buf && max_msg_len > 0) {
        pvz::api::Serializer::copy_to_c_buffer(result.message, out_msg_buf, max_msg_len);
    }
    return static_cast<int>(result.error);
}

}  // namespace

extern "C" {

int PVZ_GYM_CALL pvz_gym_get_seed_type_capacity() {
    return pvz::kSeedTypeCapacity;
}

int PVZ_GYM_CALL pvz_gym_get_seed_selection_json(char* out_json_buf, int max_len) {
    try {
        if (!out_json_buf || max_len <= 0) return -1;

        pvz::SeedSelectionState state;
        if (!pvz::seed_selection::SeedSelectionReader::read(state)) {
            return 0;
        }

        const std::string json = pvz::seed_selection::SeedSelectionSerializer::to_json(state);
        return pvz::api::Serializer::copy_to_c_buffer(json, out_json_buf, max_len);
    } catch (...) {
        return -1;
    }
}

int PVZ_GYM_CALL pvz_gym_get_seed_selection_mask(
    uint8_t* out_mask_buf, int max_size, int* out_count) {
    try {
        pvz::SeedSelectionState state;
        if (!pvz::seed_selection::SeedSelectionReader::read(state)) {
            if (out_count) *out_count = 0;
            return 0;
        }

        const auto mask = pvz::seed_selection::SeedSelectionValidator::build_action_mask(state);
        const int count = static_cast<int>(mask.size());
        if (out_count) *out_count = count;
        if (!out_mask_buf || max_size < count) return -count;

        for (int i = 0; i < count; ++i) {
            out_mask_buf[i] = mask[static_cast<std::size_t>(i)] ? 1 : 0;
        }
        return count;
    } catch (...) {
        if (out_count) *out_count = 0; return -1;
    }
}

int PVZ_GYM_CALL pvz_gym_select_seed(int plant_type, char* out_msg_buf, int max_msg_len) {
    try {
        return write_result_message(
            pvz::seed_selection::SeedSelectionExecutor::select_seed(plant_type),
            out_msg_buf, max_msg_len);
    } catch (const PvzRuntime::NativeOperationError& e) {
        return write_result_message({false, pvz::SeedSelectionError::Unsupported, e.what()}, out_msg_buf, max_msg_len);
    } catch (const std::exception& e) {
        if (out_msg_buf && max_msg_len > 0) {
            strncpy_s(out_msg_buf, max_msg_len, e.what(), max_msg_len - 1);
        }
        return -1;
    } catch (...) {
        if (out_msg_buf && max_msg_len > 0) {
            strncpy_s(out_msg_buf, max_msg_len, "unknown native exception", max_msg_len - 1);
        }
        return -1;
    }
}

int PVZ_GYM_CALL pvz_gym_select_imitater(int target_type, char* out_msg_buf, int max_msg_len) {
    try {
        return write_result_message(
            pvz::seed_selection::SeedSelectionExecutor::select_imitater(target_type),
            out_msg_buf, max_msg_len);
    } catch (const PvzRuntime::NativeOperationError& e) {
        return write_result_message({false, pvz::SeedSelectionError::Unsupported, e.what()}, out_msg_buf, max_msg_len);
    } catch (const std::exception& e) {
        if (out_msg_buf && max_msg_len > 0) {
            strncpy_s(out_msg_buf, max_msg_len, e.what(), max_msg_len - 1);
        }
        return -1;
    } catch (...) {
        if (out_msg_buf && max_msg_len > 0) {
            strncpy_s(out_msg_buf, max_msg_len, "unknown native exception", max_msg_len - 1);
        }
        return -1;
    }
}

int PVZ_GYM_CALL pvz_gym_deselect_seed(int plant_type, char* out_msg_buf, int max_msg_len) {
    try {
        return write_result_message(
            pvz::seed_selection::SeedSelectionExecutor::deselect_seed(plant_type),
            out_msg_buf, max_msg_len);
    } catch (const PvzRuntime::NativeOperationError& e) {
        return write_result_message({false, pvz::SeedSelectionError::Unsupported, e.what()}, out_msg_buf, max_msg_len);
    } catch (const std::exception& e) {
        if (out_msg_buf && max_msg_len > 0) {
            strncpy_s(out_msg_buf, max_msg_len, e.what(), max_msg_len - 1);
        }
        return -1;
    } catch (...) {
        if (out_msg_buf && max_msg_len > 0) {
            strncpy_s(out_msg_buf, max_msg_len, "unknown native exception", max_msg_len - 1);
        }
        return -1;
    }
}

int PVZ_GYM_CALL pvz_gym_confirm_seeds(char* out_msg_buf, int max_msg_len) {
    try {
        return write_result_message(
            pvz::seed_selection::SeedSelectionExecutor::confirm_seeds(),
            out_msg_buf, max_msg_len);
    } catch (const PvzRuntime::NativeOperationError& e) {
        return write_result_message({false, pvz::SeedSelectionError::Unsupported, e.what()}, out_msg_buf, max_msg_len);
    } catch (const std::exception& e) {
        if (out_msg_buf && max_msg_len > 0) {
            strncpy_s(out_msg_buf, max_msg_len, e.what(), max_msg_len - 1);
        }
        return -1;
    } catch (...) {
        if (out_msg_buf && max_msg_len > 0) {
            strncpy_s(out_msg_buf, max_msg_len, "unknown native exception", max_msg_len - 1);
        }
        return -1;
    }
}

}
