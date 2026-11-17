cmake_minimum_required(VERSION 3.24)

project(just_ba VERSION 0.1.0 DESCRIPTION "Minimal, fast bundle adjustment" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)  # decision D52; the CUDA sources are C++20 (D113)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(JBA_BUILD_WASM "Build the WebAssembly module and demo page (Emscripten only; exploration)" OFF)
if(EMSCRIPTEN)
  # Native wasm exceptions (every current browser); the library reports errors by throwing.
  add_compile_options(-fwasm-exceptions)
  add_link_options(-fwasm-exceptions)
  set(JBA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(JBA_BUILD_CLI OFF CACHE BOOL "" FORCE)
endif()
option(JBA_BUILD_TESTS "Build the tests" ON)
option(JBA_BUILD_CLI "Build the sample command-line binary" ON)
option(JBA_WITH_CERES "Build the Ceres reference harness (test-only; see third_party/ceres)" OFF)
option(JBA_BUILD_PYTHON "Build the Python bindings (needs nanobind; see README)" OFF)
# The demo app (D132): the incremental reconstruction, natively and in the browser. It
# alone links poselib-gpu, from its sources next to this repository; nothing else does.
option(JBA_BUILD_DEMO "Build the demo app (needs poselib-gpu's sources, JBA_POSELIB_GPU_DIR)" OFF)
if(JBA_BUILD_PYTHON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # static libraries linked into the extension
endif()
if(APPLE)
  option(JBA_WITH_METAL "Build the Metal backend" ON)
else()
  set(JBA_WITH_METAL OFF)
endif()
option(JBA_WITH_CUDA "Build the CUDA backend (needs the CUDA toolkit, 12 or later)" OFF)
option(JBA_WITH_WEBGPU "Build the WebGPU backend (exploration; needs Dawn natively, emdawnwebgpu under Emscripten)" OFF)
option(JBA_WITH_CUDSS "CUDA backend: sparse_schur with NVIDIA cuDSS (needs cuDSS 1.0 or later)" OFF)
if(JBA_WITH_CUDA)
  # The GPU of the build machine unless the caller lists architectures.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  # C++20, not the host's C++23: nvcc accepts C++23 only from CUDA 13.3, and the
  # CUDA 12 wheels keep the GPU generations CUDA 13 dropped (D113). The headers the
  # CUDA sources include keep to C++20.
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

# Packages looked up here, in one directory: cuDSS's package configuration promotes its
# targets to global scope each time it is read, which CMake allows only in the
# directory that made them, and a Ceres built with cuDSS (master) reads it again.
if(JBA_WITH_CUDA AND JBA_WITH_CUDSS)
  find_package(cudss REQUIRED CONFIG)
endif()
if(JBA_WITH_CERES)
  # The accuracy reference is 2.2.0 (document 05); another version, such as master
  # (2.3.0) for the GPU speed column, is named here.
  set(JBA_CERES_VERSION 2.2.0 CACHE STRING "Ceres version the reference harness is built against")
  find_package(Ceres ${JBA_CERES_VERSION} EXACT REQUIRED CONFIG)
endif()

add_library(jba_warnings INTERFACE)
if(MSVC)
  target_compile_options(jba_warnings INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /utf-8>)
else()
  target_compile_options(jba_warnings INTERFACE $<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-Wall -Wextra -Wpedantic -Wshadow>)
endif()

# Internal headers (D79): the backend interface, the trust-region loop, the geometry
# helpers and the backends' own headers. They are not in any public include
# directory; the libraries, the CLI and the tests reach them through this target.
add_library(jba_internal INTERFACE)
target_include_directories(jba_internal INTERFACE ${PROJECT_SOURCE_DIR}/core/internal
                           ${PROJECT_SOURCE_DIR}/backends/cpu/internal ${PROJECT_SOURCE_DIR}/backends/gpu/internal
                           ${PROJECT_SOURCE_DIR}/backends/metal/internal ${PROJECT_SOURCE_DIR}/backends/cuda/internal
                           ${PROJECT_SOURCE_DIR}/backends/webgpu/internal)
target_link_libraries(jba_internal INTERFACE jba_generated_cpu)  # the CPU headers include the generated code

add_subdirectory(core)
add_subdirectory(residuals)
add_subdirectory(backends/cpu)
add_subdirectory(tracks)
add_subdirectory(api)
# The GPU backends' shared host logic is plain C++: built on every platform, with or
# without a GPU backend, so every compiler CI has sees it (review of milestone 5,
# finding 7). Only a GPU backend links it.
add_subdirectory(backends/gpu)
if(JBA_WITH_METAL)
  add_subdirectory(backends/metal)
endif()
if(JBA_WITH_CUDA)
  add_subdirectory(backends/cuda)
endif()
if(JBA_WITH_WEBGPU)
  add_subdirectory(backends/webgpu)
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
if(JBA_BUILD_WASM)
  add_subdirectory(bindings/wasm)
endif()
if(JBA_BUILD_DEMO)
  add_subdirectory(demo)
endif()
if(JBA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
