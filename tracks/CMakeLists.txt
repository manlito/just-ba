# Track building and triangulation (document 04). CPU only: runs once per solve.
add_library(jba_tracks STATIC src/triangulation.cpp src/tracks.cpp)
target_include_directories(jba_tracks PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(jba_tracks PUBLIC jba_core PRIVATE jba_warnings)
