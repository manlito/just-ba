add_library(jba_cpu STATIC src/cost.cpp src/backend.cpp)
find_package(Threads REQUIRED)
target_include_directories(jba_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(jba_cpu PUBLIC jba_core jba_generated_cpu Threads::Threads PRIVATE jba_warnings)
