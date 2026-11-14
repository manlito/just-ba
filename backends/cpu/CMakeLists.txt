add_library(jba_cpu STATIC src/cost.cpp)
target_include_directories(jba_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(jba_cpu PUBLIC jba_core jba_generated_cpu PRIVATE jba_warnings)
