enable_language(OBJCXX)

# Metal sources are the generated residual header followed by the kernels, embedded
# as strings and compiled by Metal at run time (D50). No offline Metal compiler is
# needed. The solver's kernels are the GPU backends' shared source (D96) behind the
# Metal prelude.
set(_header ${PROJECT_SOURCE_DIR}/residuals/generated/metal/reprojection.metal.h)
function(jba_embed_metal symbol out)
  set(_inputs ${_header} ${ARGN})
  string(REPLACE ";" "\\;" _list "${_inputs}")
  add_custom_command(
    OUTPUT ${out}
    COMMAND ${CMAKE_COMMAND} -DOUT=${out} -DSYMBOL=${symbol}
            "-DINPUTS=${_list}" -P ${PROJECT_SOURCE_DIR}/cmake/EmbedText.cmake
    DEPENDS ${_inputs} ${PROJECT_SOURCE_DIR}/cmake/EmbedText.cmake
    COMMENT "Embedding Metal source ${symbol}")
endfunction()
set(_spike ${CMAKE_CURRENT_BINARY_DIR}/reprojection_source.cpp)
set(_solver ${CMAKE_CURRENT_BINARY_DIR}/solver_source.cpp)
jba_embed_metal(kJbaMetalReprojectionSource ${_spike} ${CMAKE_CURRENT_SOURCE_DIR}/kernels/reprojection.metal)
jba_embed_metal(kJbaMetalSolverSource ${_solver}
                ${CMAKE_CURRENT_SOURCE_DIR}/kernels/prelude.metal
                ${PROJECT_SOURCE_DIR}/backends/gpu/internal/jba/gpu/layout.h
                ${PROJECT_SOURCE_DIR}/backends/gpu/kernels/solver.inc
                ${CMAKE_CURRENT_SOURCE_DIR}/kernels/layout_sizes.metal)

add_library(jba_metal STATIC src/reprojection.mm src/backend.mm ${_spike} ${_solver})
target_compile_options(jba_metal PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
target_link_libraries(jba_metal PUBLIC jba_core jba_cpu jba_gpu "-framework Metal" "-framework Foundation" "-framework Accelerate"
                      PRIVATE jba_internal jba_warnings)
