cmake_minimum_required(VERSION 3.24)

project(just_ba VERSION 0.0.1 DESCRIPTION "Minimal, fast bundle adjustment" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)  # decision D52: the newest standard nvcc supports (CUDA 13.3+)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(JBA_BUILD_TESTS "Build the tests" ON)
option(JBA_BUILD_CLI "Build the sample command-line binary" ON)
option(JBA_WITH_CERES "Build the Ceres reference harness (test-only; see third_party/ceres)" OFF)
option(JBA_BUILD_PYTHON "Build the Python bindings (needs nanobind; see README)" OFF)
if(JBA_BUILD_PYTHON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # static libraries linked into the extension
endif()
if(APPLE)
  option(JBA_WITH_METAL "Build the Metal backend" ON)
else()
  set(JBA_WITH_METAL OFF)
endif()

add_library(jba_warnings INTERFACE)
if(MSVC)
  target_compile_options(jba_warnings INTERFACE /W4 /permissive- /utf-8)
else()
  target_compile_options(jba_warnings INTERFACE -Wall -Wextra -Wpedantic -Wshadow)
endif()

# Internal headers (D79): the backend interface, the trust-region loop, the geometry
# helpers and the backends' own headers. They are not in any public include
# directory; the libraries, the CLI and the tests reach them through this target.
add_library(jba_internal INTERFACE)
target_include_directories(jba_internal INTERFACE ${PROJECT_SOURCE_DIR}/core/internal
                           ${PROJECT_SOURCE_DIR}/backends/cpu/internal ${PROJECT_SOURCE_DIR}/backends/gpu/internal
                           ${PROJECT_SOURCE_DIR}/backends/metal/internal)
target_link_libraries(jba_internal INTERFACE jba_generated_cpu)  # the CPU headers include the generated code

add_subdirectory(core)
add_subdirectory(residuals)
add_subdirectory(backends/cpu)
add_subdirectory(tracks)
add_subdirectory(api)
if(JBA_WITH_METAL)
  add_subdirectory(backends/gpu)
  add_subdirectory(backends/metal)
endif()
if(JBA_BUILD_CLI)
  add_subdirectory(cli)
endif()
if(JBA_BUILD_TESTS)
  enable_testing()
endif()
if(JBA_BUILD_PYTHON)
  add_subdirectory(bindings/python)
endif()
if(JBA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
