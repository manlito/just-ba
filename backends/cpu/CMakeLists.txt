add_library(jba_cpu STATIC src/cost.cpp src/system.cpp src/backend.cpp)
find_package(Threads REQUIRED)
target_link_libraries(jba_cpu PUBLIC jba_core Threads::Threads PRIVATE jba_internal jba_warnings)
