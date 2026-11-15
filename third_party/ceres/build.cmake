# Builds the test-only reference solver: Eigen 3.4.0 and a minimal Ceres 2.2.0.
#
#   cmake -DPREFIX=<install dir> [-DWORK=<scratch dir>] [-DJOBS=<n>] [-DACCELERATE=ON] -P third_party/ceres/build.cmake
#
# Then configure just-ba with -DJBA_WITH_CERES=ON -DCMAKE_PREFIX_PATH=<install dir>.
#
# Minimal means Eigen only: no SuiteSparse, LAPACK, gflags, glog, CUDA or Accelerate
# (document 05). Ceres uses its bundled mini logging. Runs the same on Linux, macOS
# and Windows. Nothing here is linked into the shipped library.
#
# -DACCELERATE=ON (macOS only) builds the speed reference of milestone 4 instead:
# the same Ceres with Apple's Accelerate for sparse Cholesky (ACCELERATESPARSE) and
# LAPACK. Install it to a prefix of its own; it is used by hand, not in CI.

cmake_minimum_required(VERSION 3.24)

if(NOT PREFIX)
  message(FATAL_ERROR "set -DPREFIX=<install dir>")
endif()
get_filename_component(PREFIX "${PREFIX}" ABSOLUTE)
if(NOT WORK)
  set(WORK "${PREFIX}/_work")
endif()
if(NOT JOBS)
  cmake_host_system_information(RESULT JOBS QUERY NUMBER_OF_LOGICAL_CORES)
endif()
if(NOT CONFIG)
  set(CONFIG Release)
endif()

set(EIGEN_URL "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz")
set(EIGEN_SHA256 "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72")
set(CERES_URL "https://github.com/ceres-solver/ceres-solver/archive/refs/tags/2.2.0.tar.gz")
set(CERES_SHA256 "12efacfadbfdc1bbfa203c236e96f4d3c210bed96994288b3ff0c8e7c6f350d4")

# Options for the minimal build. Logging is limited to warnings and errors
# (miniglog levels: INFO 0, WARNING -1), so test output stays readable.
set(CERES_OPTIONS
  -DMINIGLOG=ON -DMINIGLOG_MAX_LOG_LEVEL=-1 -DGFLAGS=OFF -DSUITESPARSE=OFF -DLAPACK=OFF
  -DACCELERATESPARSE=OFF -DUSE_CUDA=OFF -DEIGENSPARSE=ON -DSCHUR_SPECIALIZATIONS=ON
  -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_DOCUMENTATION=OFF
  -DBUILD_SHARED_LIBS=OFF -DPROVIDE_UNINSTALL_TARGET=OFF)
if(ACCELERATE)
  if(NOT APPLE)
    message(FATAL_ERROR "ACCELERATE needs macOS")
  endif()
  list(TRANSFORM CERES_OPTIONS REPLACE "^-DACCELERATESPARSE=OFF$" "-DACCELERATESPARSE=ON")
  list(TRANSFORM CERES_OPTIONS REPLACE "^-DLAPACK=OFF$" "-DLAPACK=ON")
endif()

# The stamp records what was built; any change to versions or options rebuilds.
set(STAMP_FILE "${PREFIX}/.jba-ceres-stamp")
set(STAMP "${EIGEN_SHA256} ${CERES_SHA256} ${CONFIG} ${CERES_OPTIONS}")
if(EXISTS "${STAMP_FILE}")
  file(READ "${STAMP_FILE}" old_stamp)
  if(old_stamp STREQUAL STAMP)
    message(STATUS "Ceres 2.2.0 already installed in ${PREFIX}")
    return()
  endif()
endif()

file(MAKE_DIRECTORY "${WORK}")

function(fetch name url sha dir_out)
  set(archive "${WORK}/${name}.tar.gz")
  if(NOT EXISTS "${archive}")
    message(STATUS "Downloading ${url}")
    file(DOWNLOAD "${url}" "${archive}" EXPECTED_HASH SHA256=${sha} STATUS st TLS_VERIFY ON)
    list(GET st 0 code)
    if(NOT code EQUAL 0)
      file(REMOVE "${archive}")
      message(FATAL_ERROR "download of ${url} failed: ${st}")
    endif()
  endif()
  set(src "${WORK}/${name}-src")
  file(REMOVE_RECURSE "${src}")
  file(ARCHIVE_EXTRACT INPUT "${archive}" DESTINATION "${src}")
  file(GLOB top LIST_DIRECTORIES true "${src}/*")
  set(${dir_out} "${top}" PARENT_SCOPE)
endfunction()

function(run)
  execute_process(COMMAND ${ARGN} RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "command failed (${rc}): ${ARGN}")
  endif()
endfunction()

set(common
  -DCMAKE_BUILD_TYPE=${CONFIG}
  -DCMAKE_INSTALL_PREFIX=${PREFIX}
  -DCMAKE_PREFIX_PATH=${PREFIX}
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5)
if(GENERATOR)
  list(APPEND common -G "${GENERATOR}")
endif()

# Eigen: header-only, installed for its CMake config.
fetch(eigen "${EIGEN_URL}" "${EIGEN_SHA256}" eigen_src)
run(${CMAKE_COMMAND} -S "${eigen_src}" -B "${WORK}/eigen-build" ${common}
    -DBUILD_TESTING=OFF -DEIGEN_BUILD_DOC=OFF -DEIGEN_BUILD_PKGCONFIG=OFF)
run(${CMAKE_COMMAND} --build "${WORK}/eigen-build" --config ${CONFIG} --target install)

fetch(ceres "${CERES_URL}" "${CERES_SHA256}" ceres_src)
file(REMOVE_RECURSE "${WORK}/ceres-build")
run(${CMAKE_COMMAND} -S "${ceres_src}" -B "${WORK}/ceres-build" ${common} ${CERES_OPTIONS})
run(${CMAKE_COMMAND} --build "${WORK}/ceres-build" --config ${CONFIG} --parallel ${JOBS} --target install)

file(WRITE "${STAMP_FILE}" "${STAMP}")
message(STATUS "Ceres 2.2.0 installed in ${PREFIX}")
