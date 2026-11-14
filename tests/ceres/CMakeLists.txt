# Test-only reference solver. Build it with third_party/ceres/build.cmake and pass
# -DCMAKE_PREFIX_PATH=<its install dir>.
find_package(Ceres 2.2.0 EXACT REQUIRED CONFIG)

add_library(jba_ceres_reference STATIC ceres_reference.cpp)
target_link_libraries(jba_ceres_reference PUBLIC jba Ceres::ceres)

add_executable(jba_ceres_solve ceres_solve.cpp)
target_link_libraries(jba_ceres_solve PRIVATE jba_ceres_reference)
add_executable(jba_parity parity.cpp)
target_link_libraries(jba_parity PRIVATE jba_ceres_reference)
add_executable(jba_make_problem make_problem.cpp)
target_link_libraries(jba_make_problem PRIVATE jba)
target_include_directories(jba_make_problem PRIVATE ${PROJECT_SOURCE_DIR}/tests)
add_executable(test_ceres_level1 test_ceres_level1.cpp)
target_link_libraries(test_ceres_level1 PRIVATE jba_ceres_reference)
add_test(NAME test_ceres_level1 COMMAND test_ceres_level1)

# Level 2 on synthetic problems: one generated file per case, then a parity run.
#   jba_parity_case(<name> "<generator options>" "<parity options>")
# Both solvers run to tight tolerances, so they stop at the minimum rather than
# wherever the default stopping rule (|cost change| <= 1e-6 cost) fires; otherwise
# the end points could differ by about that much and the comparison would be noise.
function(jba_parity_case name gen parity)
  set(file ${CMAKE_CURRENT_BINARY_DIR}/synthetic_${name}.jba)
  separate_arguments(gen_list UNIX_COMMAND "${gen}")
  separate_arguments(parity_list UNIX_COMMAND
    "--function-tolerance 1e-12 --parameter-tolerance 1e-12 --max-iterations 200 ${parity}")
  add_test(NAME parity_make_${name} COMMAND jba_make_problem ${file} ${gen_list})
  add_test(NAME parity_${name} COMMAND jba_parity ${file} ${parity_list})
  set_tests_properties(parity_make_${name} PROPERTIES FIXTURES_SETUP synthetic_${name})
  set_tests_properties(parity_${name} PROPERTIES FIXTURES_REQUIRED synthetic_${name} LABELS parity)
endfunction()

jba_parity_case(simple_pinhole "--model SIMPLE_PINHOLE" "--check-parameters")
jba_parity_case(pinhole "--model PINHOLE" "--check-parameters")
jba_parity_case(simple_radial "--model SIMPLE_RADIAL" "--check-parameters")
jba_parity_case(radial "--model RADIAL" "--check-parameters")
jba_parity_case(no_gauge "--model RADIAL" "--gauge none")
jba_parity_case(fixed_blocks "--model PINHOLE --fix-images 2 --fix-points 20" "--check-parameters")
jba_parity_case(sigma "--model SIMPLE_RADIAL --sigma 0.7" "--check-parameters")
jba_parity_case(shared_focal "--model RADIAL --free-intrinsics 1" "--check-parameters")
jba_parity_case(shared_all_intrinsics "--model PINHOLE --free-intrinsics 7" "--check-parameters")
jba_parity_case(per_image_intrinsics "--model RADIAL --per-image-intrinsics --free-intrinsics 5" "")
jba_parity_case(huber "--model SIMPLE_RADIAL --outliers 0.05" "--loss huber:2 --check-parameters")
jba_parity_case(cauchy "--model SIMPLE_RADIAL --outliers 0.05" "--loss cauchy:2 --check-parameters")
jba_parity_case(geman_mcclure "--model SIMPLE_RADIAL --outliers 0.05" "--loss geman_mcclure:2 --check-parameters")
# Camera models added in milestone 2.
jba_parity_case(opencv "--model OPENCV" "--check-parameters")
jba_parity_case(opencv_fisheye "--model OPENCV_FISHEYE" "--check-parameters")
# With all eight parameters of one shared block free, the minimum is a flat valley:
# costs agree to about 1e-11, but parameters only to a few 1e-6, and Ceres's own
# end point moves that much between runs (its threads change the summation order).
jba_parity_case(opencv_free_distortion "--model OPENCV --free-intrinsics 7"
                "--check-parameters --max-centre-diff 2e-5 --max-rotation-diff 1e-5")
jba_parity_case(fisheye_free_distortion "--model OPENCV_FISHEYE --free-intrinsics 7"
                "--check-parameters --max-centre-diff 2e-5 --max-rotation-diff 1e-5")
# Pose priors (milestone 2). Default prior losses (Cauchy) unless stated.
jba_parity_case(relative_priors "--model PINHOLE --relative-priors 2" "--check-parameters")
jba_parity_case(relative_priors_only "--model PINHOLE --points 0 --relative-priors 3" "--check-parameters")
jba_parity_case(relative_priors_sigma "--model PINHOLE --relative-priors 2 --prior-sigma 0.01,0.02" "--check-parameters")
jba_parity_case(relative_prior_outliers "--model SIMPLE_RADIAL --relative-priors 3 --prior-outliers 0.1"
                "--check-parameters")
jba_parity_case(relative_prior_huber "--model SIMPLE_RADIAL --relative-priors 3 --prior-outliers 0.1"
                "--relative-prior-loss huber:2 --check-parameters")
jba_parity_case(relative_prior_no_loss "--model PINHOLE --relative-priors 2" "--relative-prior-loss none --check-parameters")
jba_parity_case(global_identity "--model PINHOLE --global-priors" "--check-parameters")
jba_parity_case(pose_graph "--model PINHOLE --points 0 --relative-priors 2 --global-priors --prior-outliers 0.05"
                "--check-parameters")
jba_parity_case(global_rigid "--model PINHOLE --global-priors --global-relation rigid" "--check-parameters")
jba_parity_case(global_similarity "--model PINHOLE --global-priors --global-relation similarity" "--check-parameters")
jba_parity_case(relative_and_similarity
                "--model RADIAL --relative-priors 2 --global-priors --global-relation similarity --prior-outliers 0.05"
                "--check-parameters")
jba_parity_case(priors_fixed_blocks "--model PINHOLE --relative-priors 2 --global-priors --fix-images 2"
                "--check-parameters")

# Against Ceres's exact solver: costs must agree, but an inexact inner solve takes
# more outer iterations by design, so the iteration ratio is not the matched one.
jba_parity_case(sparse_schur "--model RADIAL" "--linear-solver sparse_schur --max-iteration-ratio 3")

# Level 2 on the BAL test set, when present (fetch with tests/data/fetch_bal.py).
# Run without gauge fixing, as Ceres's own BAL example does: with the gauge held,
# venice-52's end point depends on rounding (several basins; document 05, results).
# venice-52 also needs more than 50 iterations to converge, and then agrees to about
# 2e-4; it reaches 3e-7 only after about 2,000 iterations, a manual check.
set(_bal_args_problem-52-64053-pre --max-iterations 300 --max-cost-diff 1e-3)
set(JBA_BAL_DIR "${PROJECT_SOURCE_DIR}/tests/data/bal" CACHE PATH "Directory holding downloaded BAL problems")
set(_bal_set problem-49-7776-pre problem-21-11315-pre problem-16-22106-pre problem-52-64053-pre
             problem-93-61203-pre)
set(_missing "")
foreach(_name IN LISTS _bal_set)
  set(_file "${JBA_BAL_DIR}/${_name}.txt")
  if(EXISTS "${_file}")
    add_test(NAME parity_bal_${_name} COMMAND jba_parity ${_file} --gauge none ${_bal_args_${_name}})
    set_tests_properties(parity_bal_${_name} PROPERTIES LABELS "parity;bal")
    if(_name STREQUAL "problem-52-64053-pre")
      # About four minutes with Ceres on one thread (D63); CI runs it only in full mode.
      set_tests_properties(parity_bal_${_name} PROPERTIES LABELS "parity;bal;slow")
    endif()
  else()
    list(APPEND _missing ${_name})
  endif()
endforeach()
if(_missing)
  message(STATUS "BAL problems not found in ${JBA_BAL_DIR}: ${_missing}; run tests/data/fetch_bal.py")
endif()
