cmake_minimum_required(VERSION 3.20)

# Package metadata is the single source of the native and Python version.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS pyproject.toml)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" _throwdb_version_line
    REGEX "^version = \"[0-9]+\\.[0-9]+\\.[0-9]+\"$")
if(NOT _throwdb_version_line)
    message(FATAL_ERROR "pyproject.toml must define a numeric major.minor.patch version")
endif()
string(REGEX REPLACE "^version = \"([^\"]+)\"$" "\\1"
    _throwdb_version "${_throwdb_version_line}")
project(throwdb VERSION ${_throwdb_version} LANGUAGES CXX)

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
if(THROWDB_BUILD_TESTS)
    target_sources(throwdb_core PRIVATE src/core/native_collection_test_hooks.cpp)
    target_compile_definitions(throwdb_core PRIVATE THROWDB_NATIVE_TEST_HOOKS=1)
endif()

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
            "THROWDB_BUILD_SERVER currently requires POSIX storage semantics; "
            "configure with -DTHROWDB_BUILD_SERVER=OFF on Windows")
    endif()

    add_library(throwdb_server STATIC
        src/server/collection_engine.cpp
        src/server/idempotency_journal.cpp
        src/server/read_workload_log.cpp
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
    # a daemon without its required transport when dependencies are unavailable.
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

    # gRPC is the public daemon contract.
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
    target_compile_definitions(_throwdb PRIVATE THROWDB_VERSION="${PROJECT_VERSION}")
    target_include_directories(
        _throwdb PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
    target_compile_features(_throwdb PRIVATE cxx_std_20)
    install(TARGETS _throwdb LIBRARY DESTINATION throwdb)
endif()

if(THROWDB_BUILD_TESTS)
    enable_testing()
    include(cmake/ThrowDBTests.cmake)
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
