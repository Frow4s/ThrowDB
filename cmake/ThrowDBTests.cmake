# Tests stay explicit: adding a source does not silently register a new target.
function(throwdb_add_test name library)
    set(target "throwdb_${name}_tests")
    add_executable(${target} "${PROJECT_SOURCE_DIR}/tests/${name}_tests.cpp")
    target_link_libraries(${target} PRIVATE ${library})
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_compile_definitions(${target} PRIVATE THROWDB_NATIVE_TEST_HOOKS=1)
    throwdb_enable_warnings(${target})
    add_test(NAME ${target} COMMAND ${target})
endfunction()

foreach(name IN ITEMS
    core
    format_v2
    mutation_types
    partitioned_index_format
    partitioned_index_run_storage
    partition_snapshot_storage
    partitioned_index_merge
    partitioned_index_storage
    partitioned_segment_cleaning
    partitioned_segment_storage
    partitioned_resource_inventory
    mutation_wal
    mutation_catalog
    mutation_delta
    mutation_inventory
    physical_reader
    read_scheduler
    io_uring_coordinator
    native_format
    native_collection
    native_collection_lifecycle
    native_collection_materialization
    native_collection_compaction
    native_collection_compaction_crash
    native_collection_reader_barrier
)
    throwdb_add_test(${name} throwdb::core)
endforeach()
set_tests_properties(throwdb_native_collection_compaction_crash_tests
    PROPERTIES TIMEOUT 30)

if(THROWDB_BUILD_SERVER)
    foreach(name IN ITEMS server_service server_idempotency read_workload_log)
        throwdb_add_test(${name} throwdb::server)
    endforeach()
endif()

if(THROWDB_BUILD_GRPC_SERVER)
    # A distinct, non-installed daemon carries deterministic subprocess
    # lifecycle seams. The production throwdbd target never contains
    # test-only environment variables or branches, even in test builds.
    add_executable(
        throwdbd_test_hooks src/server/grpc_main.cpp)
    target_link_libraries(
        throwdbd_test_hooks PRIVATE throwdb::grpc_transport)
    target_compile_features(
        throwdbd_test_hooks PRIVATE cxx_std_20)
    target_compile_definitions(
        throwdbd_test_hooks PRIVATE THROWDB_GRPC_TEST_HOOKS=1)
    throwdb_enable_warnings(throwdbd_test_hooks)

    add_executable(
        throwdb_grpc_server_tests tests/grpc_server_tests.cpp)
    target_link_libraries(throwdb_grpc_server_tests
        PRIVATE throwdb::grpc_transport throwdb_grpc_proto)
    target_compile_features(
        throwdb_grpc_server_tests PRIVATE cxx_std_20)
    if(THROWDB_GRPC_ENABLE_REFLECTION)
        target_compile_definitions(throwdb_grpc_server_tests
            PRIVATE THROWDB_TEST_GRPC_REFLECTION=1)
    endif()
    target_compile_definitions(throwdb_grpc_server_tests
        PRIVATE
            THROWDB_TEST_GRPC_DAEMON_PATH="$<TARGET_FILE:throwdbd_test_hooks>")
    add_dependencies(throwdb_grpc_server_tests throwdbd_test_hooks)
    throwdb_enable_warnings(throwdb_grpc_server_tests)
    add_test(
        NAME throwdb_grpc_server_tests
        COMMAND throwdb_grpc_server_tests)

    add_executable(
        throwdb_grpc_admission_tests tests/grpc_admission_tests.cpp)
    target_link_libraries(throwdb_grpc_admission_tests
        PRIVATE throwdb::grpc_transport)
    target_include_directories(throwdb_grpc_admission_tests
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/server)
    target_compile_features(
        throwdb_grpc_admission_tests PRIVATE cxx_std_20)
    throwdb_enable_warnings(throwdb_grpc_admission_tests)
    add_test(
        NAME throwdb_grpc_admission_tests
        COMMAND throwdb_grpc_admission_tests)
endif()
