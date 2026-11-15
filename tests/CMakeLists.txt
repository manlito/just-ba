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
jba_test(test_geometry)

# End to end on synthetic scenes (document 05; generator in scene.cpp).
jba_test(test_refine)
target_sources(test_refine PRIVATE scene.cpp)

if(JBA_WITH_METAL)
  jba_test(test_metal_reprojection jba_metal)
  set_tests_properties(test_metal_reprojection PROPERTIES LABELS gpu)
endif()

# Synthetic problem files, for the parity cases and the benchmark (bench/).
add_executable(jba_make_problem make_problem.cpp)
target_link_libraries(jba_make_problem PRIVATE jba jba_internal)

if(JBA_WITH_CERES)
  add_subdirectory(ceres)
endif()

# Public headers alone, without jba_internal (D79).
add_executable(test_public_headers test_public_headers.cpp)
target_link_libraries(test_public_headers PRIVATE jba jba_warnings)
add_test(NAME test_public_headers COMMAND test_public_headers)
