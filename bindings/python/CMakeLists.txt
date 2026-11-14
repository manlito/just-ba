# Python bindings (nanobind, stable ABI from Python 3.12). Configure with the
# interpreter that has nanobind installed, for example:
#   -DJBA_BUILD_PYTHON=ON -DPython_EXECUTABLE=bindings/python/.venv/bin/python
# The package is assembled in <build>/bindings/python/just_ba; the tests run from there.
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT RESULT_VARIABLE _nb)
if(NOT _nb EQUAL 0)
  message(FATAL_ERROR "nanobind not found for ${Python_EXECUTABLE}; install it (see README) or set Python_EXECUTABLE")
endif()
find_package(nanobind CONFIG REQUIRED)

set(_pkg ${CMAKE_CURRENT_BINARY_DIR}/just_ba)
nanobind_add_module(_core STABLE_ABI NB_STATIC src/module.cpp)
target_link_libraries(_core PRIVATE jba)
set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${_pkg})
foreach(_cfg ${CMAKE_CONFIGURATION_TYPES})
  string(TOUPPER ${_cfg} _CFG)
  set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${_CFG} ${_pkg})
endforeach()

file(GLOB _py CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/just_ba/*.py)
foreach(_f ${_py})
  get_filename_component(_n ${_f} NAME)
  add_custom_command(TARGET _core POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_f} ${_pkg}/${_n})
endforeach()

if(JBA_BUILD_TESTS)
  add_test(NAME python_tests COMMAND ${Python_EXECUTABLE} -m pytest -q ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  set_tests_properties(python_tests PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}" LABELS python)
endif()
