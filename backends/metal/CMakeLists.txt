enable_language(OBJCXX)

# The Metal source is the generated residual header followed by the kernels, embedded
# as a string and compiled by Metal at run time. No offline Metal compiler is needed.
set(_header ${PROJECT_SOURCE_DIR}/residuals/generated/metal/reprojection.metal.h)
set(_kernels ${CMAKE_CURRENT_SOURCE_DIR}/kernels/reprojection.metal)
set(_embedded ${CMAKE_CURRENT_BINARY_DIR}/reprojection_source.cpp)
add_custom_command(
  OUTPUT ${_embedded}
  COMMAND ${CMAKE_COMMAND} -DOUT=${_embedded} -DSYMBOL=kJbaMetalReprojectionSource
          "-DINPUTS=${_header}\;${_kernels}" -P ${PROJECT_SOURCE_DIR}/cmake/EmbedText.cmake
  DEPENDS ${_header} ${_kernels} ${PROJECT_SOURCE_DIR}/cmake/EmbedText.cmake
  COMMENT "Embedding Metal reprojection source")

add_library(jba_metal STATIC src/reprojection.mm ${_embedded})
target_include_directories(jba_metal PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(jba_metal PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
target_link_libraries(jba_metal PUBLIC jba_core "-framework Metal" "-framework Foundation"
                      PRIVATE jba_warnings)
