# The library's public entry points, above core and the backends (document 03):
# solve, the residual report, and the refine pipeline. The C-level API will live
# here too.
add_library(jba STATIC src/solve.cpp src/report.cpp src/refine.cpp src/covariance.cpp src/export.cpp)
target_include_directories(jba PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(jba PUBLIC jba_core jba_cpu jba_tracks PRIVATE jba_internal jba_warnings)
if(JBA_WITH_METAL)
  target_link_libraries(jba PUBLIC jba_metal)
  target_compile_definitions(jba PRIVATE JBA_HAVE_METAL=1)
endif()
if(JBA_WITH_CUDA)
  target_link_libraries(jba PUBLIC jba_cuda)
  target_compile_definitions(jba PRIVATE JBA_HAVE_CUDA=1)
endif()
