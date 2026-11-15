# WebGPU backend (exploration). Natively against Dawn's prebuilt library
# (JBA_DAWN_DIR, the unpacked release with lib/cmake/Dawn, lib64/ on Linux); under Emscripten against
# Dawn's emdawnwebgpu port (JBA_EMDAWNWEBGPU_DIR, the unpacked emdawnwebgpu_pkg).
# The kernels are WGSL, embedded as strings and compiled by the WebGPU implementation
# at run time, like Metal's (D50).

function(jba_embed_wgsl symbol out input)
  add_custom_command(
    OUTPUT ${out}
    COMMAND ${CMAKE_COMMAND} -DOUT=${out} -DSYMBOL=${symbol} "-DINPUTS=${input}" -P
            ${PROJECT_SOURCE_DIR}/cmake/EmbedText.cmake
    DEPENDS ${input} ${PROJECT_SOURCE_DIR}/cmake/EmbedText.cmake
    COMMENT "Embedding WGSL source ${symbol}")
endfunction()
set(_gen ${CMAKE_CURRENT_BINARY_DIR})
jba_embed_wgsl(kJbaWgslReprojection ${_gen}/wgsl_reprojection.cpp
               ${PROJECT_SOURCE_DIR}/residuals/generated/wgsl/reprojection.wgsl)
jba_embed_wgsl(kJbaWgslSolver ${_gen}/wgsl_solver.cpp ${CMAKE_CURRENT_SOURCE_DIR}/kernels/solver.wgsl)
jba_embed_wgsl(kJbaWgslLinearize ${_gen}/wgsl_linearize.cpp ${CMAKE_CURRENT_SOURCE_DIR}/kernels/linearize.wgsl)

add_library(jba_webgpu STATIC src/backend.cpp ${_gen}/wgsl_reprojection.cpp ${_gen}/wgsl_solver.cpp
                              ${_gen}/wgsl_linearize.cpp)
target_include_directories(jba_webgpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/internal)
target_link_libraries(jba_webgpu PUBLIC jba_core jba_cpu jba_gpu PRIVATE jba_internal jba_warnings)

if(EMSCRIPTEN)
  set(JBA_EMDAWNWEBGPU_DIR "${PROJECT_SOURCE_DIR}/third_party/webgpu/emdawnwebgpu/emdawnwebgpu_pkg"
      CACHE PATH "The unpacked emdawnwebgpu_pkg of a Dawn release")
  set(_port "--use-port=${JBA_EMDAWNWEBGPU_DIR}/emdawnwebgpu.port.py")
  target_compile_options(jba_webgpu PUBLIC ${_port})
  target_link_options(jba_webgpu PUBLIC ${_port})
else()
  file(GLOB _dawn_default "${PROJECT_SOURCE_DIR}/third_party/webgpu/dawn/Dawn-*")
  set(JBA_DAWN_DIR "${_dawn_default}" CACHE PATH "The unpacked Dawn release (with lib/cmake/Dawn)")
  # Dawn's Linux release installs into lib64/ (CI's first WebGPU run), macOS's into lib/.
  find_package(Dawn REQUIRED CONFIG PATHS "${JBA_DAWN_DIR}/lib/cmake/Dawn" "${JBA_DAWN_DIR}/lib64/cmake/Dawn"
               NO_DEFAULT_PATH)
  find_package(Threads REQUIRED)
  target_link_libraries(jba_webgpu PUBLIC dawn::webgpu_dawn)
  if(APPLE)
    target_link_libraries(jba_webgpu PUBLIC "-framework Accelerate")
  endif()
endif()
