cmake_minimum_required(VERSION 3.20)

project(throwdb VERSION 0.1.0 LANGUAGES CXX)

find_package(Threads REQUIRED)

option(THROWDB_BUILD_PYTHON "Build the Python extension" ON)
option(THROWDB_BUILD_TESTS "Build the C++ tests" ON)
option(THROWDB_BUILD_BENCHMARKS "Build benchmark tools" ON)

add_library(throwdb_core STATIC
    src/core/format.cpp
    src/core/index.cpp
    src/core/storage_reader.cpp
    src/core/storage_writer.cpp
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

if(MSVC)
    target_compile_options(throwdb_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(throwdb_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(THROWDB_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_throwdb MODULE bindings/python_module.cpp)
    target_link_libraries(_throwdb PRIVATE throwdb::core)
    target_compile_features(_throwdb PRIVATE cxx_std_20)
    install(TARGETS _throwdb LIBRARY DESTINATION throwdb)
endif()

if(THROWDB_BUILD_TESTS)
    enable_testing()
    add_executable(throwdb_core_tests tests/core_tests.cpp)
    target_link_libraries(throwdb_core_tests PRIVATE throwdb::core)
    target_compile_features(throwdb_core_tests PRIVATE cxx_std_20)
    add_test(NAME throwdb_core_tests COMMAND throwdb_core_tests)
endif()

if(THROWDB_BUILD_BENCHMARKS)
    add_executable(throwdb_bench benchmarks/throwdb_bench.cpp)
    target_link_libraries(throwdb_bench PRIVATE throwdb::core)
    target_compile_features(throwdb_bench PRIVATE cxx_std_20)
endif()

include(GNUInstallDirs)
install(TARGETS throwdb_core
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
