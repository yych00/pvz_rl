cmake_minimum_required(VERSION 3.20)
project(pvz_gym_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PVZ_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/vendor/pvztoolkit")
set(MINIZ_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/vendor/miniz")
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

add_library(pvz_gym_native SHARED
    ${PVZ_ROOT}/src/process.cpp
    ${PVZ_ROOT}/src/code.cpp
    ${PVZ_ROOT}/src/data.cpp
    ${PVZ_ROOT}/src/pvz.cpp
    ${PVZ_ROOT}/src/lineup.cpp
    ${PVZ_ROOT}/src/utils.cpp

    src/process/process_manager.cpp
    src/process/memory_io.cpp
    src/process/runtime.cpp
    src/process/runtime_injection.cpp
    src/process/main_loop_barrier.cpp
    src/game/simulation_clock.cpp
    src/game/board_update_batch.cpp
    src/game/board_update_batch_injector.cpp
    src/game/headless_clock.cpp
    src/game/runtime_phase.cpp
    src/game/card_sources.cpp
    src/game/rng_control.cpp
    src/game/level.cpp

    src/game/original/plant_calls.cpp
    src/game/original/shovel_calls.cpp
    src/game/original/cob_calls.cpp
    src/game/original/vase_calls.cpp
    src/game/rules/plant_rules.cpp
    src/game/rules/shovel_rules.cpp
    src/game/rules/cob_rules.cpp
    src/action/action_codec.cpp
    src/action/action_resolver.cpp
    src/action/action_validator.cpp
    src/action/action_mask.cpp
    src/action/action_executor.cpp

    src/observation/reader_global.cpp
    src/observation/reader_cards.cpp
    src/observation/reader_plants.cpp
    src/observation/reader_zombies.cpp
    src/observation/zombie_decoder.cpp
    src/observation/reader_dropped_cards.cpp
    src/observation/reader_grid_items.cpp
    src/observation/reader_ice_trails.cpp
    src/observation/reader_mowers.cpp
    src/observation/reader_grid.cpp
    src/observation/observation_builder.cpp
    src/api/serializer.cpp
    src/api/binary_serializer.cpp
    src/api/shared_result.cpp
    src/api/seed_selection_c_api.cpp
    src/api/speed_c_api.cpp
    src/api/runtime_c_api.cpp
    src/runtime/turbo_runtime.cpp
    src/runtime/resident_executor.cpp
    src/runtime/native_runtime.cpp
    src/runtime/audio_mute.cpp
    src/seed_selection/seed_selection_reader.cpp
    src/seed_selection/seed_selection_validator.cpp
    src/seed_selection/seed_selection_executor.cpp
    src/seed_selection/seed_selection_serializer.cpp
    src/automation/auto_collect.cpp
    src/automation/auto_whack.cpp
    src/automation/dialogue_helper.cpp
    src/automation/tutorial_skipper.cpp
    src/automation/intro_skipper.cpp
    src/automation/automation_controller.cpp
    src/env/events_tracker.cpp
    src/env/pvz_env.cpp
    src/api/c_api.cpp

    ${MINIZ_ROOT}/miniz.c
    ${MINIZ_ROOT}/miniz_tdef.c
    ${MINIZ_ROOT}/miniz_tinfl.c
    ${MINIZ_ROOT}/miniz_zip.c
)

target_include_directories(pvz_gym_native PRIVATE
    include
    ${PVZ_ROOT}/inc
    ${MINIZ_ROOT}
)

target_compile_definitions(pvz_gym_native PRIVATE
    UNICODE
    _UNICODE
    WIN32_LEAN_AND_MEAN
    NOMINMAX
    _WIN32_WINNT=0x0600
    _REGEX_MAX_STACK_COUNT=0
    _PVZ_BETA_LEAK_SUPPORT
)

if(MSVC)
    target_compile_options(pvz_gym_native PRIVATE
        /utf-8
        $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
        "$<$<COMPILE_LANGUAGE:CXX>:/FI${CMAKE_CURRENT_SOURCE_DIR}/include/runtime/compat_x64.h>"
    )
endif()

target_link_libraries(pvz_gym_native PRIVATE
    crypt32
    advapi32
    wintrust
    shlwapi
    user32
    gdi32
    ole32
    uuid
)

set_target_properties(pvz_gym_native PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/out"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/out"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/out"
)

# Offline fact-contract tests: no PlantsVsZombies.exe process is required.
enable_testing()
add_executable(pvz_gym_runtime_framework_test
    tests/runtime_framework_test.cpp
    src/api/serializer.cpp
)
target_link_libraries(pvz_gym_runtime_framework_test PRIVATE pvz_gym_native)
add_custom_command(TARGET pvz_gym_runtime_framework_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:pvz_gym_native>
        $<TARGET_FILE_DIR:pvz_gym_runtime_framework_test>)
target_include_directories(pvz_gym_runtime_framework_test PRIVATE include)
if(MSVC)
    target_compile_options(pvz_gym_runtime_framework_test PRIVATE /utf-8)
endif()
add_test(NAME runtime_framework COMMAND pvz_gym_runtime_framework_test)
add_executable(pvz_gym_interaction_contract_test
    tests/interaction_contract_test.cpp
    src/api/serializer.cpp
)
target_include_directories(pvz_gym_interaction_contract_test PRIVATE include)
add_test(NAME interaction_contract COMMAND pvz_gym_interaction_contract_test)
add_executable(pvz_gym_thread_desktop_test tests/thread_desktop_test.cpp)
target_link_libraries(pvz_gym_thread_desktop_test PRIVATE user32)
add_test(NAME thread_desktop COMMAND pvz_gym_thread_desktop_test)
add_executable(pvz_gym_events_tracker_test
    tests/events_tracker_test.cpp
    src/env/events_tracker.cpp
)
target_include_directories(pvz_gym_events_tracker_test PRIVATE include)
add_test(NAME events_tracker COMMAND pvz_gym_events_tracker_test)

add_executable(pvz_gym_action_codec_test
    tests/action_codec_test.cpp
    src/action/action_codec.cpp
)
target_include_directories(pvz_gym_action_codec_test PRIVATE include)
add_test(NAME action_codec COMMAND pvz_gym_action_codec_test)

add_executable(pvz_gym_zombie_facts_test
    tests/zombie_facts_test.cpp
    src/observation/zombie_decoder.cpp
    src/api/serializer.cpp
)
target_include_directories(pvz_gym_zombie_facts_test PRIVATE include)
if(MSVC)
    target_compile_options(pvz_gym_zombie_facts_test PRIVATE /utf-8)
endif()
add_test(NAME zombie_facts COMMAND pvz_gym_zombie_facts_test)

# Execute generated x86 instructions in real threads, without a game process.
# Configure a Win32 build to run these even when the production DLL is x64.
if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    add_executable(pvz_gym_resident_executor_test tests/resident_executor_test.cpp)
    target_compile_definitions(pvz_gym_resident_executor_test PRIVATE NOMINMAX)
    add_test(NAME resident_executor COMMAND pvz_gym_resident_executor_test)
    add_executable(pvz_gym_level_reset_code_test
        tests/level_reset_code_test.cpp
        ${PVZ_ROOT}/src/code.cpp
    )
    target_include_directories(pvz_gym_level_reset_code_test PRIVATE ${PVZ_ROOT}/inc)
    target_compile_definitions(pvz_gym_level_reset_code_test PRIVATE NOMINMAX)
    if(MSVC)
        target_compile_options(pvz_gym_level_reset_code_test PRIVATE /utf-8)
    endif()
    add_test(NAME level_reset_code COMMAND pvz_gym_level_reset_code_test)
    add_executable(pvz_gym_board_update_batch_test
        tests/board_update_batch_test.cpp
        src/game/board_update_batch_injector.cpp
        ${PVZ_ROOT}/src/code.cpp
    )
    target_include_directories(pvz_gym_board_update_batch_test PRIVATE ${PVZ_ROOT}/inc)
    target_compile_definitions(pvz_gym_board_update_batch_test PRIVATE NOMINMAX)
    add_test(NAME board_update_batch COMMAND pvz_gym_board_update_batch_test)

    add_executable(pvz_gym_thread_safe_point_test tests/thread_safe_point_test.cpp)
    add_test(NAME thread_safe_point COMMAND pvz_gym_thread_safe_point_test)
elseif(WIN32)
    add_executable(pvz_gym_wow64_safe_point_test tests/wow64_safe_point_test.cpp)
    set(PVZ_GYM_X86_TEST_FIXTURE "" CACHE FILEPATH "Win32 thread-safe-point test executable")
    if(PVZ_GYM_X86_TEST_FIXTURE)
        add_test(NAME wow64_safe_point COMMAND pvz_gym_wow64_safe_point_test
            "${PVZ_GYM_X86_TEST_FIXTURE}")
    endif()
endif()

add_executable(pvz_gym_checked_memory_test tests/checked_memory_test.cpp)
target_include_directories(pvz_gym_checked_memory_test PRIVATE include)
add_test(NAME checked_memory COMMAND pvz_gym_checked_memory_test)

add_executable(pvz_gym_boss_observation_contract_test
    tests/boss_observation_contract_test.cpp
    src/env/events_tracker.cpp
    src/api/serializer.cpp
    src/api/binary_serializer.cpp
)
target_include_directories(pvz_gym_boss_observation_contract_test PRIVATE include)
if(MSVC)
    target_compile_options(pvz_gym_boss_observation_contract_test PRIVATE /utf-8)
endif()
add_test(NAME boss_observation_contract COMMAND pvz_gym_boss_observation_contract_test)
