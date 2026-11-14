# The library's public entry points, above core and the backends (document 03):
# solve, the residual report, and the refine pipeline. The C-level API will live
# here too.
add_library(jba STATIC src/solve.cpp src/report.cpp src/refine.cpp)
target_include_directories(jba PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(jba PUBLIC jba_core jba_cpu jba_tracks PRIVATE jba_warnings)
