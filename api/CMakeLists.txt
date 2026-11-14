# The library's public entry points, above core and the backends (document 03).
# The C-level API will live here too.
add_library(jba STATIC src/solve.cpp)
target_link_libraries(jba PUBLIC jba_core jba_cpu PRIVATE jba_warnings)
