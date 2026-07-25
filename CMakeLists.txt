cmake_minimum_required(VERSION 3.20)

project(throwdb VERSION 0.1.0 LANGUAGES CXX)

find_package(Threads REQUIRED)

option(THROWDB_BUILD_PYTHON "Build the Python extension" ON)
option(THROWDB_BUILD_TESTS "Build the C++ tests" ON)
option(THROWDB_BUILD_BENCHMARKS "Build benchmark tools" ON)
option(THROWDB_BUILD_SERVER "Build the single-collection C++ server" ON)
option(
    THROWDB_BUILD_GRPC_SERVER
    "Build the first-class C++ gRPC single-collection transport"
    OFF)
option(
    THROWDB_GRPC_ENABLE_REFLECTION
    "Expose the gRPC server reflection service for grpcurl/diagnostics"
    ON)
option(
    THROWDB_PACKAGE_NATIVE_SERVER
    "Install the gRPC daemon as an internal Python-package executable"
    OFF)
set(
    THROWDB_GRPC_PREFIX
    "$ENV{THROWDB_GRPC_PREFIX}"
    CACHE PATH
    "Private prefix containing the pinned gRPC and Protobuf CMake packages")
option(THROWDB_INSTALL_CPP_SDK "Install the C++ static library and headers" ON)

add_library(throwdb_core STATIC
    src/core/crc32c.cpp
    src/core/file_lock.cpp
    src/core/format.cpp
    src/core/index.cpp
    src/core/io_uring_coordinator.cpp
    src/core/io_uring_engine.cpp
    src/core/io_uring_pool.cpp
    src/core/mutation_catalog.cpp
    src/core/mutation_delta.cpp
    src/core/mutation_inventory.cpp
    src/core/mutation_wal.cpp
    src/core/native_collection.cpp
    src/core/native_collection_test_hooks.cpp
    src/core/native_format.cpp
    src/core/partition_snapshot_storage.cpp
    src/core/partitioned_index_format.cpp
    src/core/partitioned_index_merge.cpp
    src/core/partitioned_index_run_storage.cpp
    src/core/partitioned_index_storage.cpp
    src/core/partitioned_resource_inventory.cpp
    src/core/partitioned_segment_storage.cpp
    src/core/physical_reader.cpp
    src/core/read_scheduler.cpp
    src/core/sha256.cpp
    src/core/storage_reader.cpp
    src/core/storage_writer.cpp
    src/core/wal_file.cpp
)
add_library(throwdb::core ALIAS throwdb_core)

target_link_libraries(throwdb_core PUBLIC Threads::Threads)

target_compile_features(throwdb_core PUBLIC cxx_std_20)
target_include_directories(throwdb_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
)
set_target_properties(throwdb_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

function(throwdb_enable_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

throwdb_enable_warnings(throwdb_core)

if(THROWDB_BUILD_SERVER)
    if(WIN32)
        message(FATAL_ERROR
            "THROWDB_BUILD_SERVER currently requires POSIX sockets; "
            "configure with -DTHROWDB_BUILD_SERVER=OFF on Windows")
    endif()

    add_library(throwdb_server STATIC
        src/server/collection_engine.cpp
        src/server/idempotency_journal.cpp
        src/server/protocol.cpp
        src/server/read_workload_log.cpp
        src/server/single_collection_service.cpp
        src/server/tcp_server.cpp
    )
    add_library(throwdb::server ALIAS throwdb_server)
    target_link_libraries(throwdb_server PUBLIC throwdb::core)
    target_compile_features(throwdb_server PUBLIC cxx_std_20)
    target_include_directories(throwdb_server
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core
            ${CMAKE_CURRENT_SOURCE_DIR}/src/server
    )
    set_target_properties(
        throwdb_server PROPERTIES POSITION_INDEPENDENT_CODE ON)
    throwdb_enable_warnings(throwdb_server)

    add_executable(throwdbd_tcp src/server/main.cpp)
    set_target_properties(throwdbd_tcp PROPERTIES OUTPUT_NAME throwdbd-tcp)
    target_link_libraries(throwdbd_tcp PRIVATE throwdb::server)
    target_compile_features(throwdbd_tcp PRIVATE cxx_std_20)
    target_include_directories(
        throwdbd_tcp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/server)
    throwdb_enable_warnings(throwdbd_tcp)
endif()

if(THROWDB_BUILD_GRPC_SERVER)
    if(NOT THROWDB_BUILD_SERVER)
        message(FATAL_ERROR
            "THROWDB_BUILD_GRPC_SERVER=ON requires "
            "THROWDB_BUILD_SERVER=ON")
    endif()

    # bootstrap_grpc.sh and source-package builds share this project-specific
    # hint. This keeps pip/scikit-build invocations independent of the more
    # general CMAKE_PREFIX_PATH environment variable while still allowing an
    # explicit -DTHROWDB_GRPC_PREFIX=... cache value to take precedence. Read
    # the environment again when an older build tree cached an empty value.
    set(_throwdb_grpc_dependency_prefix "${THROWDB_GRPC_PREFIX}")
    if(NOT _throwdb_grpc_dependency_prefix
       AND DEFINED ENV{THROWDB_GRPC_PREFIX})
        set(_throwdb_grpc_dependency_prefix
            "$ENV{THROWDB_GRPC_PREFIX}")
    endif()
    if(_throwdb_grpc_dependency_prefix)
        list(PREPEND CMAKE_PREFIX_PATH
            "${_throwdb_grpc_dependency_prefix}")
        message(STATUS
            "Using ThrowDB gRPC dependency prefix: "
            "${_throwdb_grpc_dependency_prefix}")
    endif()

    # Deliberately REQUIRED: requesting the gRPC transport must never produce
    # a TCP-only binary because a dependency happened to be unavailable.
    find_package(Protobuf CONFIG REQUIRED)
    find_package(gRPC CONFIG REQUIRED)
    unset(_throwdb_grpc_dependency_prefix)
    if(THROWDB_GRPC_ENABLE_REFLECTION
       AND NOT TARGET gRPC::grpc++_reflection)
        message(FATAL_ERROR
            "THROWDB_GRPC_ENABLE_REFLECTION=ON requires the imported "
            "target gRPC::grpc++_reflection")
    endif()

    set(THROWDB_GRPC_PROTO
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/throwdb/v1/throwdb.proto)
    set(THROWDB_GRPC_GENERATED_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/generated/grpc)
    set(THROWDB_GRPC_PROTO_CC
        ${THROWDB_GRPC_GENERATED_DIR}/throwdb/v1/throwdb.pb.cc)
    set(THROWDB_GRPC_PROTO_H
        ${THROWDB_GRPC_GENERATED_DIR}/throwdb/v1/throwdb.pb.h)
    set(THROWDB_GRPC_SERVICE_CC
        ${THROWDB_GRPC_GENERATED_DIR}/throwdb/v1/throwdb.grpc.pb.cc)
    set(THROWDB_GRPC_SERVICE_H
        ${THROWDB_GRPC_GENERATED_DIR}/throwdb/v1/throwdb.grpc.pb.h)

    add_custom_command(
        OUTPUT
            ${THROWDB_GRPC_PROTO_CC}
            ${THROWDB_GRPC_PROTO_H}
            ${THROWDB_GRPC_SERVICE_CC}
            ${THROWDB_GRPC_SERVICE_H}
        COMMAND ${CMAKE_COMMAND} -E make_directory
            ${THROWDB_GRPC_GENERATED_DIR}
        COMMAND $<TARGET_FILE:protobuf::protoc>
        ARGS
            --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto
            --cpp_out=${THROWDB_GRPC_GENERATED_DIR}
            --grpc_out=${THROWDB_GRPC_GENERATED_DIR}
            --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
            ${THROWDB_GRPC_PROTO}
        DEPENDS
            ${THROWDB_GRPC_PROTO}
            protobuf::protoc
            gRPC::grpc_cpp_plugin
        COMMENT "Generating ThrowDB gRPC/protobuf sources"
        VERBATIM)

    add_library(throwdb_grpc_proto STATIC
        ${THROWDB_GRPC_PROTO_CC}
        ${THROWDB_GRPC_SERVICE_CC})
    target_link_libraries(throwdb_grpc_proto
        PUBLIC protobuf::libprotobuf gRPC::grpc++)
    target_include_directories(throwdb_grpc_proto
        PUBLIC $<BUILD_INTERFACE:${THROWDB_GRPC_GENERATED_DIR}>)
    target_compile_features(throwdb_grpc_proto PUBLIC cxx_std_20)
    set_target_properties(
        throwdb_grpc_proto PROPERTIES POSITION_INDEPENDENT_CODE ON)
    throwdb_enable_warnings(throwdb_grpc_proto)

    add_library(throwdb_grpc_transport STATIC
        src/server/grpc_admission.cpp
        src/server/grpc_server.cpp)
    add_library(throwdb::grpc_transport ALIAS throwdb_grpc_transport)
    target_link_libraries(throwdb_grpc_transport
        PUBLIC throwdb::server
        PRIVATE throwdb_grpc_proto)
    if(THROWDB_GRPC_ENABLE_REFLECTION)
        target_link_libraries(throwdb_grpc_transport
            PRIVATE gRPC::grpc++_reflection)
        target_compile_definitions(throwdb_grpc_transport
            PRIVATE THROWDB_GRPC_ENABLE_REFLECTION=1)
    endif()
    target_include_directories(throwdb_grpc_transport
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/server)
    target_compile_features(throwdb_grpc_transport PUBLIC cxx_std_20)
    set_target_properties(
        throwdb_grpc_transport PROPERTIES POSITION_INDEPENDENT_CODE ON)
    throwdb_enable_warnings(throwdb_grpc_transport)

    # gRPC is the public daemon contract. The provisional custom transport is
    # intentionally named throwdbd-tcp so clients cannot mistake its endpoint
    # for gRPC.
    add_executable(throwdbd src/server/grpc_main.cpp)
    if(THROWDB_PACKAGE_NATIVE_SERVER)
        # The wheel's PATH-level `throwdbd` is the one-shot Python resolver.
        # A distinct internal filename prevents recursive launcher execution.
        set_target_properties(
            throwdbd PROPERTIES OUTPUT_NAME _throwdbd_native)
    endif()
    target_link_libraries(throwdbd PRIVATE throwdb::grpc_transport)
    target_compile_features(throwdbd PRIVATE cxx_std_20)
    throwdb_enable_warnings(throwdbd)
endif()

if(THROWDB_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_throwdb MODULE bindings/python_module.cpp)
    target_link_libraries(_throwdb PRIVATE throwdb::core)
    target_include_directories(
        _throwdb PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
    target_compile_features(_throwdb PRIVATE cxx_std_20)
    install(TARGETS _throwdb LIBRARY DESTINATION throwdb)
endif()

if(THROWDB_BUILD_TESTS)
    enable_testing()
    add_executable(throwdb_core_tests tests/core_tests.cpp)
    target_link_libraries(throwdb_core_tests PRIVATE throwdb::core)
    target_compile_features(throwdb_core_tests PRIVATE cxx_std_20)
    throwdb_enable_warnings(throwdb_core_tests)
    add_test(NAME throwdb_core_tests COMMAND throwdb_core_tests)

    add_executable(throwdb_format_v2_tests tests/format_v2_tests.cpp)
    target_link_libraries(throwdb_format_v2_tests PRIVATE throwdb::core)
    target_compile_features(throwdb_format_v2_tests PRIVATE cxx_std_20)
    throwdb_enable_warnings(throwdb_format_v2_tests)
    add_test(NAME throwdb_format_v2_tests COMMAND throwdb_format_v2_tests)

    add_executable(throwdb_mutation_types_tests tests/mutation_types_tests.cpp)
    target_link_libraries(throwdb_mutation_types_tests PRIVATE throwdb::core)
    target_compile_features(throwdb_mutation_types_tests PRIVATE cxx_std_20)
    throwdb_enable_warnings(throwdb_mutation_types_tests)
    add_test(NAME throwdb_mutation_types_tests COMMAND throwdb_mutation_types_tests)

    add_executable(
        throwdb_partitioned_index_format_tests
        tests/partitioned_index_format_tests.cpp)
    target_link_libraries(
        throwdb_partitioned_index_format_tests PRIVATE throwdb::core)
    target_compile_features(
        throwdb_partitioned_index_format_tests PRIVATE cxx_std_20)
    throwdb_enable_warnings(throwdb_partitioned_index_format_tests)
    add_test(
        NAME throwdb_partitioned_index_format_tests
        COMMAND throwdb_partitioned_index_format_tests)

    add_executable(
        throwdb_partitioned_index_run_storage_tests
        tests/partitioned_index_run_storage_tests.cpp)
    target_link_libraries(
        throwdb_partitioned_index_run_storage_tests PRIVATE throwdb::core)
    target_compile_features(
        throwdb_partitioned_index_run_storage_tests PRIVATE cxx_std_20)
    throwdb_enable_warnings(throwdb_partitioned_index_run_storage_tests)
    add_test(
        NAME throwdb_partitioned_index_run_storage_tests
        COMMAND throwdb_partitioned_index_run_storage_tests)

    add_executable(
        throwdb_partition_snapshot_storage_tests
        tests/partition_snapshot_storage_tests.cpp)
    target_link_libraries(
        throwdb_partition_snapshot_storage_tests PRIVATE throwdb::core)
    target_compile_features(
        throwdb_partition_snapshot_storage_tests PRIVATE cxx_std_20)
    throwdb_enable_warnings(throwdb_partition_snapshot_storage_tests)
    add_test(
        NAME throwdb_partition_snapshot_storage_tests
        COMMAND throwdb_partition_snapshot_storage_tests)

    add_executable(
        throwdb_partitioned_index_merge_tests
        tests/partitioned_index_merge_tests.cpp)
    target_link_libraries(
        throwdb_partitioned_index_merge_tests PRIVATE throwdb::core)
    target_compile_features(
        throwdb_partitioned_index_merge_tests PRIVATE cxx_std_20)
    throwdb_enable_warnings(throwdb_partitioned_index_merge_tests)
    add_test(
        NAME throwdb_partitioned_index_merge_tests
        COMMAND throwdb_partitioned_index_merge_tests)

    add_executable(
        throwdb_partitioned_index_storage_tests
        tests/partitioned_index_storage_tests.cpp)
    target_link_libraries(
        throwdb_partitioned_index_storage_tests PRIVATE throwdb::core)
    target_compile_features(
        throwdb_partitioned_index_storage_tests PRIVATE cxx_std_20)
    throwdb_enable_warnings(throwdb_partitioned_index_storage_tests)
    add_test(
        NAME throwdb_partitioned_index_storage_tests
        COMMAND throwdb_partitioned_index_storage_tests)

    add_executable(
        throwdb_partitioned_segment_cleaning_tests
        tests/partitioned_segment_cleaning_tests.cpp)
    target_link_libraries(
        throwdb_partitioned_segment_cleaning_tests PRIVATE throwdb::core)
    target_compile_features(
        throwdb_partitioned_segment_cleaning_tests PRIVATE cxx_std_20)
    throwdb_enable_warnings(throwdb_partitioned_segment_cleaning_tests)
    add_test(
        NAME throwdb_partitioned_segment_cleaning_tests
        COMMAND throwdb_partitioned_segment_cleaning_tests)

    add_executable(
        throwdb_partitioned_segment_storage_tests
        tests/partitioned_segment_storage_tests.cpp)
    target_link_libraries(
        throwdb_partitioned_segment_storage_tests PRIVATE throwdb::core)
    target_compile_features(
        throwdb_partitioned_segment_storage_tests PRIVATE cxx_std_20)
    throwdb_enable_warnings(throwdb_partitioned_segment_storage_tests)
    add_test(
        NAME throwdb_partitioned_segment_storage_tests
        COMMAND throwdb_partitioned_segment_storage_tests)

    add_executable(
        throwdb_partitioned_resource_inventory_tests
        tests/partitioned_resource_inventory_tests.cpp)
    target_link_libraries(
        throwdb_partitioned_resource_inventory_tests PRIVATE throwdb::core)
    target_compile_features(
        throwdb_partitioned_resource_inventory_tests PRIVATE cxx_std_20)
    throwdb_enable_warnings(throwdb_partitioned_resource_inventory_tests)
    add_test(
        NAME throwdb_partitioned_resource_inventory_tests
        COMMAND throwdb_partitioned_resource_inventory_tests)

    add_executable(throwdb_mutation_wal_tests tests/mutation_wal_tests.cpp)
    target_link_libraries(throwdb_mutation_wal_tests PRIVATE throwdb::core)
    target_compile_features(throwdb_mutation_wal_tests PRIVATE cxx_std_20)
    throwdb_enable_warnings(throwdb_mutation_wal_tests)
    add_test(NAME throwdb_mutation_wal_tests COMMAND throwdb_mutation_wal_tests)

    add_executable(
        throwdb_mutation_catalog_tests tests/mutation_catalog_tests.cpp)
    target_link_libraries(throwdb_mutation_catalog_tests PRIVATE throwdb::core)
    target_compile_features(
        throwdb_mutation_catalog_tests PRIVATE cxx_std_20)
    throwdb_enable_warnings(throwdb_mutation_catalog_tests)
    add_test(
        NAME throwdb_mutation_catalog_tests
        COMMAND throwdb_mutation_catalog_tests)

    add_executable(
        throwdb_mutation_delta_tests tests/mutation_delta_tests.cpp)
    target_link_libraries(throwdb_mutation_delta_tests PRIVATE throwdb::core)
    target_compile_features(
        throwdb_mutation_delta_tests PRIVATE cxx_std_20)
    throwdb_enable_warnings(throwdb_mutation_delta_tests)
    add_test(
        NAME throwdb_mutation_delta_tests
        COMMAND throwdb_mutation_delta_tests)

    add_executable(
        throwdb_mutation_inventory_tests tests/mutation_inventory_tests.cpp)
    target_link_libraries(
        throwdb_mutation_inventory_tests PRIVATE throwdb::core)
    target_compile_features(
        throwdb_mutation_inventory_tests PRIVATE cxx_std_20)
    throwdb_enable_warnings(throwdb_mutation_inventory_tests)
    add_test(
        NAME throwdb_mutation_inventory_tests
        COMMAND throwdb_mutation_inventory_tests)

    add_executable(throwdb_physical_reader_tests tests/physical_reader_tests.cpp)
    target_link_libraries(throwdb_physical_reader_tests PRIVATE throwdb::core)
    target_compile_features(throwdb_physical_reader_tests PRIVATE cxx_std_20)
    throwdb_enable_warnings(throwdb_physical_reader_tests)
    add_test(NAME throwdb_physical_reader_tests COMMAND throwdb_physical_reader_tests)

    add_executable(
        throwdb_read_scheduler_tests tests/read_scheduler_tests.cpp)
    target_link_libraries(
        throwdb_read_scheduler_tests PRIVATE throwdb::core)
    target_compile_features(
        throwdb_read_scheduler_tests PRIVATE cxx_std_20)
    throwdb_enable_warnings(throwdb_read_scheduler_tests)
    add_test(
        NAME throwdb_read_scheduler_tests
        COMMAND throwdb_read_scheduler_tests)

    add_executable(
        throwdb_io_uring_coordinator_tests
        tests/io_uring_coordinator_tests.cpp)
    target_link_libraries(
        throwdb_io_uring_coordinator_tests PRIVATE throwdb::core)
    target_compile_features(
        throwdb_io_uring_coordinator_tests PRIVATE cxx_std_20)
    throwdb_enable_warnings(throwdb_io_uring_coordinator_tests)
    add_test(
        NAME throwdb_io_uring_coordinator_tests
        COMMAND throwdb_io_uring_coordinator_tests)

    add_executable(throwdb_native_format_tests tests/native_format_tests.cpp)
    target_link_libraries(throwdb_native_format_tests PRIVATE throwdb::core)
    target_compile_features(throwdb_native_format_tests PRIVATE cxx_std_20)
    throwdb_enable_warnings(throwdb_native_format_tests)
    add_test(NAME throwdb_native_format_tests COMMAND throwdb_native_format_tests)

    add_executable(throwdb_native_collection_tests tests/native_collection_tests.cpp)
    target_link_libraries(throwdb_native_collection_tests PRIVATE throwdb::core)
    target_compile_features(throwdb_native_collection_tests PRIVATE cxx_std_20)
    throwdb_enable_warnings(throwdb_native_collection_tests)
    add_test(NAME throwdb_native_collection_tests COMMAND throwdb_native_collection_tests)

    add_executable(
        throwdb_native_collection_lifecycle_tests
        tests/native_collection_lifecycle_tests.cpp)
    target_link_libraries(
        throwdb_native_collection_lifecycle_tests PRIVATE throwdb::core)
    target_compile_features(
        throwdb_native_collection_lifecycle_tests PRIVATE cxx_std_20)
    throwdb_enable_warnings(throwdb_native_collection_lifecycle_tests)
    add_test(
        NAME throwdb_native_collection_lifecycle_tests
        COMMAND throwdb_native_collection_lifecycle_tests)

    add_executable(
        throwdb_native_collection_materialization_tests
        tests/native_collection_materialization_tests.cpp)
    target_link_libraries(
        throwdb_native_collection_materialization_tests PRIVATE throwdb::core)
    target_compile_features(
        throwdb_native_collection_materialization_tests PRIVATE cxx_std_20)
    throwdb_enable_warnings(
        throwdb_native_collection_materialization_tests)
    add_test(
        NAME throwdb_native_collection_materialization_tests
        COMMAND throwdb_native_collection_materialization_tests)

    add_executable(
        throwdb_native_collection_compaction_tests
        tests/native_collection_compaction_tests.cpp)
    target_link_libraries(
        throwdb_native_collection_compaction_tests PRIVATE throwdb::core)
    target_compile_features(
        throwdb_native_collection_compaction_tests PRIVATE cxx_std_20)
    throwdb_enable_warnings(
        throwdb_native_collection_compaction_tests)
    add_test(
        NAME throwdb_native_collection_compaction_tests
        COMMAND throwdb_native_collection_compaction_tests)

    add_executable(
        throwdb_native_collection_compaction_crash_tests
        tests/native_collection_compaction_crash_tests.cpp)
    target_link_libraries(
        throwdb_native_collection_compaction_crash_tests PRIVATE throwdb::core)
    target_compile_features(
        throwdb_native_collection_compaction_crash_tests PRIVATE cxx_std_20)
    throwdb_enable_warnings(
        throwdb_native_collection_compaction_crash_tests)
    add_test(
        NAME throwdb_native_collection_compaction_crash_tests
        COMMAND throwdb_native_collection_compaction_crash_tests)
    set_tests_properties(
        throwdb_native_collection_compaction_crash_tests
        PROPERTIES TIMEOUT 30)

    add_executable(
        throwdb_native_collection_reader_barrier_tests
        tests/native_collection_reader_barrier_tests.cpp)
    target_link_libraries(
        throwdb_native_collection_reader_barrier_tests PRIVATE throwdb::core)
    target_compile_features(
        throwdb_native_collection_reader_barrier_tests PRIVATE cxx_std_20)
    throwdb_enable_warnings(
        throwdb_native_collection_reader_barrier_tests)
    add_test(
        NAME throwdb_native_collection_reader_barrier_tests
        COMMAND throwdb_native_collection_reader_barrier_tests)

    if(THROWDB_BUILD_SERVER)
        add_executable(
            throwdb_server_protocol_tests tests/server_protocol_tests.cpp)
        target_link_libraries(
            throwdb_server_protocol_tests PRIVATE throwdb::server)
        target_compile_features(
            throwdb_server_protocol_tests PRIVATE cxx_std_20)
        throwdb_enable_warnings(throwdb_server_protocol_tests)
        add_test(
            NAME throwdb_server_protocol_tests
            COMMAND throwdb_server_protocol_tests)

        add_executable(
            throwdb_server_service_tests tests/server_service_tests.cpp)
        target_link_libraries(
            throwdb_server_service_tests PRIVATE throwdb::server)
        target_compile_features(
            throwdb_server_service_tests PRIVATE cxx_std_20)
        throwdb_enable_warnings(throwdb_server_service_tests)
        add_test(
            NAME throwdb_server_service_tests
            COMMAND throwdb_server_service_tests)

        add_executable(
            throwdb_server_idempotency_tests
            tests/server_idempotency_tests.cpp)
        target_link_libraries(
            throwdb_server_idempotency_tests PRIVATE throwdb::server)
        target_compile_features(
            throwdb_server_idempotency_tests PRIVATE cxx_std_20)
        throwdb_enable_warnings(throwdb_server_idempotency_tests)
        add_test(
            NAME throwdb_server_idempotency_tests
            COMMAND throwdb_server_idempotency_tests)

        add_executable(
            throwdb_read_workload_log_tests
            tests/read_workload_log_tests.cpp)
        target_link_libraries(
            throwdb_read_workload_log_tests PRIVATE throwdb::server)
        target_compile_features(
            throwdb_read_workload_log_tests PRIVATE cxx_std_20)
        throwdb_enable_warnings(throwdb_read_workload_log_tests)
        add_test(
            NAME throwdb_read_workload_log_tests
            COMMAND throwdb_read_workload_log_tests)
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
endif()

if(THROWDB_BUILD_BENCHMARKS)
    add_executable(throwdb_bench benchmarks/throwdb_bench.cpp)
    target_link_libraries(throwdb_bench PRIVATE throwdb::core)
    target_compile_features(throwdb_bench PRIVATE cxx_std_20)

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_executable(throwdb_io_uring_probe benchmarks/io_uring_probe.cpp)
        target_compile_features(throwdb_io_uring_probe PRIVATE cxx_std_20)
    endif()
endif()

include(GNUInstallDirs)
if(THROWDB_INSTALL_CPP_SDK)
    install(TARGETS throwdb_core
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
    # The transport headers are not a stable SDK yet and their corresponding
    # static libraries/generated protobuf headers are intentionally private to
    # the daemon build. Do not publish a partial, unlinkable server SDK.
    install(
        DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        PATTERN "throwdb/server" EXCLUDE)
endif()

if(THROWDB_BUILD_GRPC_SERVER)
    # A wheel exposes a PATH-level Python launcher. The native process stays
    # private and is only reached after tune resolution. Standalone CMake
    # installs retain the conventional native bin/throwdbd entry point.
    if(THROWDB_PACKAGE_NATIVE_SERVER AND DEFINED SKBUILD_PLATLIB_DIR)
        install(
            TARGETS throwdbd
            RUNTIME DESTINATION
                "${SKBUILD_PLATLIB_DIR}/throwdb/libexec")
        install(
            FILES proto/throwdb/v1/throwdb.proto
            DESTINATION
                "${SKBUILD_PLATLIB_DIR}/throwdb/proto/throwdb/v1")
    else()
        install(
            TARGETS throwdbd
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
        install(
            FILES proto/throwdb/v1/throwdb.proto
            DESTINATION
                ${CMAKE_INSTALL_DATADIR}/throwdb/proto/throwdb/v1)
    endif()
endif()
