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
#   demo/tools/fetch_match_web.sh   # keypoints and matcher, for the page
#   cmake --build build/demo-wasm   # build/demo-wasm/demo/web: the page and everything it loads

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
    "-sJSPI_EXPORTS=demo_reconstruct,demo_backends,demo_session_new,demo_session_add,demo_session_retry,demo_session_finish,demo_capture_reconstruct"
    "-sEXPORTED_FUNCTIONS=_demo_reconstruct,_demo_backends,_demo_points,_demo_tracks,_demo_colour_obs,_malloc,_free,\
_demo_session_new,_demo_session_add,_demo_session_retry,_demo_session_snapshot,_demo_session_finish,_demo_session_free,\
_demo_pair_inliers,_demo_last_error,_demo_capture_reconstruct"
    "-sEXPORTED_RUNTIME_METHODS=ccall,HEAPU8,HEAP32,HEAPF32,HEAPF64,UTF8ToString"
    -sMODULARIZE
    -sEXPORT_ES6
    -sEXPORT_NAME=createDemo
    # The live capture's session runs in a worker (D157).
    -sENVIRONMENT=web,worker
    -sALLOW_MEMORY_GROWTH
    -sMAXIMUM_MEMORY=4GB
    -sSTACK_SIZE=2MB)
  # The page, its script and sample images, and the keypoints and matcher modules
  # (demo/tools/fetch_match_web.sh) it loads from jkp/ and jmt/ (D156).
  set(JBA_DEMO_MATCH_WEB_DIR "${PROJECT_SOURCE_DIR}/third_party/match-web" CACHE PATH
      "just-keypoints' and just-matching's browser modules, for the live demo")
  if(NOT EXISTS "${JBA_DEMO_MATCH_WEB_DIR}/jkp/jkp.js" OR NOT EXISTS "${JBA_DEMO_MATCH_WEB_DIR}/jmt/jmt.js")
    message(WARNING "No keypoints and matcher modules in ${JBA_DEMO_MATCH_WEB_DIR}: the live demo needs them "
                    "(demo/tools/fetch_match_web.sh)")
  endif()
  set(_web ${CMAKE_CURRENT_BINARY_DIR}/web)
  add_custom_command(
    TARGET jba_demo_wasm POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/web/index.html
            ${CMAKE_CURRENT_SOURCE_DIR}/web/demo.js ${CMAKE_CURRENT_SOURCE_DIR}/web/capture.html
            ${CMAKE_CURRENT_SOURCE_DIR}/web/capture.js ${CMAKE_CURRENT_SOURCE_DIR}/web/capture_worker.js
            ${CMAKE_CURRENT_SOURCE_DIR}/web/weights_worker.js ${CMAKE_CURRENT_SOURCE_DIR}/web/flexglue_weights.js
            ${CMAKE_CURRENT_SOURCE_DIR}/web/gates.js
            ${CMAKE_CURRENT_SOURCE_DIR}/web/view.js ${_web}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/web/samples ${_web}/samples)
  if(EXISTS "${JBA_DEMO_MATCH_WEB_DIR}")
    add_custom_command(TARGET jba_demo_wasm POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory
                                                               ${JBA_DEMO_MATCH_WEB_DIR} ${_web})
  endif()
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
    # D158: images placed from their pair when none sees enough of the map; the scenes still pass.
    add_test(NAME demo_synth_pairs COMMAND jba_demo synth --register-from-pairs)
    # D166: a block per image with autofocus jitter, a calibration 5 % off, the blocks in
    # one group of the intrinsics prior; the scenes still pass, and grouped the focal
    # lengths land nearer the truth than refined one by one.
    add_test(NAME demo_synth_groups COMMAND jba_demo synth --focal-jitter 0.004 --focal-bias 0.05 --groups)
    add_executable(test_demo_groups tests/test_groups.cpp)
    target_link_libraries(test_demo_groups PRIVATE jba_demo_mapper jba_warnings)
    add_test(NAME demo_groups COMMAND test_demo_groups)
    # The progress callback (D146): continuing changes nothing, returning false cancels.
    add_executable(test_demo_cancel tests/test_cancel.cpp)
    target_link_libraries(test_demo_cancel PRIVATE jba_demo_mapper jba_warnings)
    add_test(NAME demo_cancel COMMAND test_demo_cancel)
    add_executable(test_demo_diagnostics tests/test_diagnostics.cpp)
    target_link_libraries(test_demo_diagnostics PRIVATE jba_demo_mapper jba_warnings)
    add_test(NAME demo_diagnostics COMMAND test_demo_diagnostics)
    # Sessions (D153): the map grown image by image, then finished from the live map.
    add_executable(test_demo_session tests/test_session.cpp)
    target_link_libraries(test_demo_session PRIVATE jba_demo_mapper jba_warnings)
    add_test(NAME demo_session COMMAND test_demo_session)
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
