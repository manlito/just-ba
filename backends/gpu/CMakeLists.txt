# The GPU backends' shared host logic (D96): plain C++ above a Device. The kernels it
# dispatches are in kernels/solver.inc, compiled by each backend.
add_library(jba_gpu STATIC src/backend.cpp)
target_include_directories(jba_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/internal)
target_link_libraries(jba_gpu PUBLIC jba_core jba_cpu PRIVATE jba_internal jba_warnings)
