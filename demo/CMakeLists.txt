# The demo app (D134): a minimal incremental reconstruction (document 04, part C) with
# poselib-gpu's relative pose and PnP and this library's tracks, triangulation and
# solver. Natively, a command-line binary (generated scenes against their ground
# truth, and the comparison with COLMAP); under Emscripten, a WebAssembly module and
# the page of the live demo. Only this directory links poselib-gpu.
#
#   cmake -S . -B build/demo -G Ninja -DJBA_BUILD_DEMO=ON
#   cmake --build build/demo --target jba_demo && build/demo/demo/jba_demo synth
#
#   emcmake cmake -S . -B build/demo-wasm -G Ninja -DJBA_BUILD_DEMO=ON -DJBA_WITH_WEBGPU=ON \
#     -DJBA_EMDAWNWEBGPU_DIR=<emdawnwebgpu_pkg>
#   cmake --build build/demo-wasm   # build/demo-wasm/demo/web: index.html, jba_demo.js, jba_demo.wasm

set(JBA_POSELIB_GPU_DIR "${PROJECT_SOURCE_DIR}/../poselib-gpu" CACHE PATH "poselib-gpu sources, for the demo app")
if(NOT EXISTS "${JBA_POSELIB_GPU_DIR}/CMakeLists.txt" OR NOT EXISTS "${JBA_POSELIB_GPU_DIR}/include/jpo/jpo.h")
  message(FATAL_ERROR "JBA_BUILD_DEMO needs poselib-gpu's sources; none at ${JBA_POSELIB_GPU_DIR} "
                      "(set JBA_POSELIB_GPU_DIR)")
endif()

# poselib-gpu's library only: no command line, tests, test pages or PoseLib reference.
set(JPO_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(JPO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(JPO_BUILD_WASM OFF CACHE BOOL "" FORCE)
set(JPO_REFERENCE OFF CACHE STRING "" FORCE)
# Its GPU backends follow this build's: the same GPUs, and for WebGPU the same Dawn.
if(JBA_WITH_METAL)
  set(JPO_METAL AUTO CACHE STRING "" FORCE)
else()
  set(JPO_METAL OFF CACHE STRING "" FORCE)
endif()
if(JBA_WITH_CUDA)
  set(JPO_CUDA ON CACHE STRING "" FORCE)
else()
  set(JPO_CUDA OFF CACHE STRING "" FORCE)
endif()
if(JBA_WITH_WEBGPU)
  set(JPO_WEBGPU ON CACHE STRING "" FORCE)
  if(EMSCRIPTEN)
    set(JPO_EMDAWNWEBGPU_DIR "${JBA_EMDAWNWEBGPU_DIR}" CACHE PATH "" FORCE)
  else()
    set(JPO_DAWN_DIR "${JBA_DAWN_DIR}" CACHE PATH "" FORCE)
  endif()
else()
  set(JPO_WEBGPU OFF CACHE STRING "" FORCE)
endif()
add_subdirectory("${JBA_POSELIB_GPU_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/poselib-gpu" EXCLUDE_FROM_ALL)

# The mapper, the generated scenes and the comparison with their ground truth.
add_library(jba_demo_mapper STATIC src/mapper.cpp src/scenes.cpp src/evaluate.cpp src/align.cpp src/container.cpp
                                   src/capture_io.cpp src/compare.cpp)
target_include_directories(jba_demo_mapper PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(jba_demo_mapper PUBLIC jpo jba PRIVATE jba_warnings)

if(EMSCRIPTEN)
  # The live demo: one module with both libraries, single-threaded, waiting for the
  # GPU through JSPI (D124), next to its page.
  add_executable(jba_demo_wasm web/demo_wasm.cpp)
  target_link_libraries(jba_demo_wasm PRIVATE jba_demo_mapper jba_warnings)
  set_target_properties(jba_demo_wasm PROPERTIES OUTPUT_NAME jba_demo SUFFIX ".js" RUNTIME_OUTPUT_DIRECTORY
                                                 "${CMAKE_CURRENT_BINARY_DIR}/web")
  target_link_options(
    jba_demo_wasm PRIVATE
    -sJSPI
    "-sJSPI_EXPORTS=demo_run,demo_backends"
    "-sEXPORTED_FUNCTIONS=_demo_run,_demo_backends"
    "-sEXPORTED_RUNTIME_METHODS=ccall"
    -sMODULARIZE
    -sEXPORT_ES6
    -sEXPORT_NAME=createDemo
    -sENVIRONMENT=web
    -sALLOW_MEMORY_GROWTH
    -sMAXIMUM_MEMORY=4GB
    -sSTACK_SIZE=2MB)
  add_custom_command(TARGET jba_demo_wasm POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${CMAKE_CURRENT_SOURCE_DIR}/web/index.html ${CMAKE_CURRENT_BINARY_DIR}/web/index.html)
else()
  add_executable(jba_demo src/main.cpp)
  target_link_libraries(jba_demo PRIVATE jba_demo_mapper jba_warnings)
  if(JBA_BUILD_TESTS)
    # The three generated scenes reconstruct within their thresholds (CPU).
    add_test(NAME demo_synth COMMAND jba_demo synth)
    # Focal lengths 2.2 times too large and marked as a guess, as COLMAP's on images
    # without metadata: searched for, then refined (D144).
    add_test(NAME demo_synth_focal COMMAND jba_demo synth --focal-error 1.2)
    add_test(NAME demo_synth_priors COMMAND jba_demo synth --priors 0.002)
    # The progress callback (D146): continuing changes nothing, returning false cancels.
    add_executable(test_demo_cancel tests/test_cancel.cpp)
    target_link_libraries(test_demo_cancel PRIVATE jba_demo_mapper jba_warnings)
    add_test(NAME demo_cancel COMMAND test_demo_cancel)
    add_executable(test_demo_diagnostics tests/test_diagnostics.cpp)
    target_link_libraries(test_demo_diagnostics PRIVATE jba_demo_mapper jba_warnings)
    add_test(NAME demo_diagnostics COMMAND test_demo_diagnostics)
    # The COLMAP path without COLMAP: scene -> database and text model -> capture file ->
    # reconstruction against the truth (demo/tools/test_colmap_roundtrip.py).
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
      add_test(NAME demo_colmap_roundtrip
               COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tools/test_colmap_roundtrip.py
                       $<TARGET_FILE:jba_demo> ${CMAKE_CURRENT_BINARY_DIR}/colmap-roundtrip)
    endif()
  endif()
endif()
