enable_language(OBJCXX)

# Metal sources are the generated residual header followed by the kernels, embedded
# as strings and compiled by Metal at run time (D50). No offline Metal compiler is
# needed.
set(_header ${PROJECT_SOURCE_DIR}/residuals/generated/metal/reprojection.metal.h)
function(jba_embed_metal kernels symbol out)
  add_custom_command(
    OUTPUT ${out}
    COMMAND ${CMAKE_COMMAND} -DOUT=${out} -DSYMBOL=${symbol}
            "-DINPUTS=${_header}\;${kernels}" -P ${PROJECT_SOURCE_DIR}/cmake/EmbedText.cmake
    DEPENDS ${_header} ${kernels} ${PROJECT_SOURCE_DIR}/cmake/EmbedText.cmake
    COMMENT "Embedding Metal source ${symbol}")
endfunction()
set(_spike ${CMAKE_CURRENT_BINARY_DIR}/reprojection_source.cpp)
set(_solver ${CMAKE_CURRENT_BINARY_DIR}/solver_source.cpp)
jba_embed_metal(${CMAKE_CURRENT_SOURCE_DIR}/kernels/reprojection.metal kJbaMetalReprojectionSource ${_spike})
jba_embed_metal(${CMAKE_CURRENT_SOURCE_DIR}/kernels/solver.metal kJbaMetalSolverSource ${_solver})

add_library(jba_metal STATIC src/reprojection.mm src/backend.mm ${_spike} ${_solver})
target_compile_options(jba_metal PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
target_link_libraries(jba_metal PUBLIC jba_core jba_cpu "-framework Metal" "-framework Foundation"
                      PRIVATE jba_internal jba_warnings)
