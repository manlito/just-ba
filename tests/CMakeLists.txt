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

if(JBA_WITH_METAL)
  jba_test(test_metal_reprojection jba_metal)
  set_tests_properties(test_metal_reprojection PROPERTIES LABELS gpu)

  # The Metal backend against the CPU reference (milestone 4), on the parity cases'
  # problems (tests/ceres) and the BAL test set when present. Label gpu.
  add_executable(jba_metal_check metal_check.cpp)
  target_link_libraries(jba_metal_check PRIVATE jba jba_internal jba_warnings)
  function(jba_metal_case name gen check)
    set(file ${CMAKE_CURRENT_BINARY_DIR}/metal_${name}.jba)
    separate_arguments(gen_list UNIX_COMMAND "${gen}")
    separate_arguments(check_list UNIX_COMMAND "${check}")
    add_test(NAME metal_make_${name} COMMAND jba_make_problem ${file} ${gen_list})
    add_test(NAME metal_${name} COMMAND jba_metal_check ${file} ${check_list})
    set_tests_properties(metal_make_${name} PROPERTIES FIXTURES_SETUP metal_${name})
    set_tests_properties(metal_${name} PROPERTIES FIXTURES_REQUIRED metal_${name} LABELS gpu SKIP_RETURN_CODE 77)
  endfunction()
  include(metal_cases.cmake)
endif()

if(JBA_WITH_CERES)
  add_subdirectory(ceres)
endif()

# Public headers alone, without jba_internal (D79).
add_executable(test_public_headers test_public_headers.cpp)
target_link_libraries(test_public_headers PRIVATE jba jba_warnings)
add_test(NAME test_public_headers COMMAND test_public_headers)
