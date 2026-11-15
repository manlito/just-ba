add_library(jba_core STATIC
  src/camera_model.cpp
  src/initialize.cpp
  src/priors.cpp
  src/problem_io.cpp
  src/solver.cpp
  src/trust_region.cpp)
target_include_directories(jba_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(jba_core PRIVATE jba_internal jba_warnings)
