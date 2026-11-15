function(jba_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE jba jba_internal jba_warnings ${ARGN})
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

jba_test(test_reprojection_jacobians)
jba_test(test_problem_io)
jba_test(test_bal)
jba_test(test_solver)
jba_test(test_prior_jacobians)
jba_test(test_direction_jacobians)
jba_test(test_covariance)
jba_test(test_export)
jba_test(test_geometry)

# End to end on synthetic scenes (document 05; generator in scene.cpp).
jba_test(test_refine)
target_sources(test_refine PRIVATE scene.cpp)
jba_test(test_manhattan)
target_sources(test_manhattan PRIVATE scene.cpp)

# Synthetic problem files, for the parity cases and the benchmark (bench/).
add_executable(jba_make_problem make_problem.cpp)
target_link_libraries(jba_make_problem PRIVATE jba jba_internal)

# A problem and its COLMAP text model, for the comparison with Caspar (milestone 5).
add_executable(jba_to_colmap to_colmap.cpp)
target_link_libraries(jba_to_colmap PRIVATE jba jba_internal jba_warnings)

if(JBA_WITH_METAL)
  jba_test(test_metal_reprojection jba_metal)
  set_tests_properties(test_metal_reprojection PROPERTIES LABELS gpu)
endif()

# The GPU backends against the CPU reference (milestones 4 and 5), on the parity
# cases' problems (tests/ceres) and a few of their own (gpu_cases.cmake), with each
# linear solver a backend has. Tests <backend>_<case> and <backend>_sparse_<case>,
# label gpu; skipped without a device.
set(_gpu_backends)
set(_gpu_sparse_backends)  # those with sparse_schur
if(JBA_WITH_METAL)
  list(APPEND _gpu_backends metal)
  list(APPEND _gpu_sparse_backends metal)
endif()
if(JBA_WITH_CUDA)
  list(APPEND _gpu_backends cuda)
  if(JBA_WITH_CUDSS)
    list(APPEND _gpu_sparse_backends cuda)
  endif()
endif()
if(_gpu_backends)
  add_executable(jba_gpu_check gpu_check.cpp)
  target_link_libraries(jba_gpu_check PRIVATE jba jba_internal jba_warnings)
  # The test hooks of jba/gpu/backend.h; the library itself comes with jba.
  target_include_directories(jba_gpu_check PRIVATE $<TARGET_PROPERTY:jba_gpu,INTERFACE_INCLUDE_DIRECTORIES>)
  # `solvers`: "both", or "sparse" for a case that only sparse_schur can take.
  function(_jba_gpu_tests name gen check solvers)
    set(file ${CMAKE_CURRENT_BINARY_DIR}/gpu_${name}.jba)
    separate_arguments(gen_list UNIX_COMMAND "${gen}")
    separate_arguments(check_list UNIX_COMMAND "${check}")
    add_test(NAME gpu_make_${name} COMMAND jba_make_problem ${file} ${gen_list})
    set_tests_properties(gpu_make_${name} PROPERTIES FIXTURES_SETUP gpu_${name})
    foreach(be IN LISTS _gpu_backends)
      set(tests)
      if(solvers STREQUAL "both")
        add_test(NAME ${be}_${name} COMMAND jba_gpu_check ${file} --backend ${be} ${check_list})
        list(APPEND tests ${be}_${name})
      endif()
      if(be IN_LIST _gpu_sparse_backends)
        add_test(NAME ${be}_sparse_${name} COMMAND jba_gpu_check ${file} --backend ${be} ${check_list}
                                                  --linear-solver sparse_schur)
        list(APPEND tests ${be}_sparse_${name})
      endif()
      if(tests)
        set_tests_properties(${tests} PROPERTIES FIXTURES_REQUIRED gpu_${name} LABELS gpu SKIP_RETURN_CODE 77)
      endif()
    endforeach()
  endfunction()
  function(jba_gpu_case name gen check)
    _jba_gpu_tests(${name} "${gen}" "${check}" both)
  endfunction()
  function(jba_gpu_sparse_case name gen check)
    _jba_gpu_tests(${name} "${gen}" "${check}" sparse)
  endfunction()
  # A check of the backends with sparse_schur only, on an existing case's file:
  # <backend>_<name>.
  function(jba_gpu_sparse_check name case check)
    separate_arguments(check_list UNIX_COMMAND "${check}")
    foreach(be IN LISTS _gpu_sparse_backends)
      add_test(NAME ${be}_${name} COMMAND jba_gpu_check ${CMAKE_CURRENT_BINARY_DIR}/gpu_${case}.jba --backend ${be}
                                          ${check_list})
      set_tests_properties(${be}_${name} PROPERTIES FIXTURES_REQUIRED gpu_${case} LABELS gpu SKIP_RETURN_CODE 77)
    endforeach()
  endfunction()
  include(gpu_cases.cmake)
endif()

# The CLI's --warm-up and the summary fields of D101 (document 15, finding 8).
if(TARGET just-ba)
  add_test(NAME cli_make_problem COMMAND jba_make_problem ${CMAKE_CURRENT_BINARY_DIR}/cli_problem.jba --images 8 --points 200)
  set_tests_properties(cli_make_problem PROPERTIES FIXTURES_SETUP cli_problem)
  add_test(NAME cli_warm_up COMMAND just-ba solve ${CMAKE_CURRENT_BINARY_DIR}/cli_problem.jba --warm-up)
  set_tests_properties(cli_warm_up PROPERTIES FIXTURES_REQUIRED cli_problem
                       PASS_REGULAR_EXPRESSION "\"setup_seconds\": [0-9.]+, \"warm_up\": true, \"linear_solver_fallbacks\": 0")
endif()

if(JBA_WITH_CERES)
  add_subdirectory(ceres)
endif()

# Public headers alone, without jba_internal (D79).
add_executable(test_public_headers test_public_headers.cpp)
target_link_libraries(test_public_headers PRIVATE jba jba_warnings)
add_test(NAME test_public_headers COMMAND test_public_headers)
