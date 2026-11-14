# Test-only reference solver. Build it with third_party/ceres/build.cmake and pass
# -DCMAKE_PREFIX_PATH=<its install dir>.
find_package(Ceres 2.2.0 EXACT REQUIRED CONFIG)

add_executable(jba_ceres_solve ceres_solve.cpp)
target_link_libraries(jba_ceres_solve PRIVATE jba_cpu Ceres::ceres)

# A synthetic problem written by the CLI's own tests is not available here, so the
# always-on Ceres test converts a tiny generated BAL file; see make_tiny_problem.
add_executable(jba_make_tiny_problem make_tiny_problem.cpp)
target_link_libraries(jba_make_tiny_problem PRIVATE jba_cpu)
target_include_directories(jba_make_tiny_problem PRIVATE ${PROJECT_SOURCE_DIR}/tests)

add_test(NAME ceres_tiny_make COMMAND jba_make_tiny_problem ${CMAKE_CURRENT_BINARY_DIR}/tiny.jba)
add_test(NAME ceres_tiny_solve COMMAND jba_ceres_solve ${CMAKE_CURRENT_BINARY_DIR}/tiny.jba --linear-solver dense_schur)
set_tests_properties(ceres_tiny_make PROPERTIES FIXTURES_SETUP tiny_problem)
set_tests_properties(ceres_tiny_solve PROPERTIES FIXTURES_REQUIRED tiny_problem)

# A real BAL problem, when present (fetch with tests/data/fetch_bal.py).
set(JBA_BAL_DIR "${PROJECT_SOURCE_DIR}/tests/data/bal" CACHE PATH "Directory holding downloaded BAL problems")
set(_ladybug "${JBA_BAL_DIR}/problem-49-7776-pre.txt")
if(EXISTS "${_ladybug}")
  add_test(NAME ceres_bal_ladybug_49 COMMAND jba_ceres_solve ${_ladybug} --linear-solver sparse_schur)
else()
  message(STATUS "BAL ladybug-49 not found in ${JBA_BAL_DIR}; run tests/data/fetch_bal.py to enable its Ceres test")
endif()
