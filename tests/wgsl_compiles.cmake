# The WebGPU backend's WGSL compiles (D132): the whole module, as the backend hands it
# to the device, through Tint's command-line tool from the Dawn release the build
# uses. Tint resolves it (a WGSL error fails here), then translates it to every shader
# language this Tint can write of MSL, SPIR-V and HLSL, and runs the validators it has.
# No GPU needed, so CI runs it.
#
#   cmake -DWRITER=<jba_wgsl_module> -DTINT=<tint> -DOUT=<dir> -P wgsl_compiles.cmake

foreach(v WRITER TINT OUT)
  if(NOT DEFINED ${v})
    message(FATAL_ERROR "wgsl_compiles.cmake: -D${v}=... missing")
  endif()
endforeach()
file(MAKE_DIRECTORY ${OUT})
set(module ${OUT}/module.wgsl)

execute_process(COMMAND ${WRITER} ${module} RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "jba_wgsl_module failed (${rc})")
endif()

# The output formats this Tint was built with, from its help text: "--format <none|wgsl|...>".
execute_process(COMMAND ${TINT} --help OUTPUT_VARIABLE help ERROR_VARIABLE help)
string(REGEX MATCH "--format <([a-z|]+)>" _ "${help}")
string(REPLACE "|" ";" formats "${CMAKE_MATCH_1}")
if(NOT formats)
  message(FATAL_ERROR "could not read Tint's output formats from its help text")
endif()

# WGSL: parsed, resolved, through Tint's IR and written back.
set(targets wgsl)
foreach(f msl spirv hlsl)
  if(f IN_LIST formats)
    list(APPEND targets ${f})
  endif()
endforeach()
message(STATUS "Tint writes: ${formats}; checking ${targets}")

# To standard output: Tint writes every entry point there, but only one to a file.
foreach(f IN LISTS targets)
  execute_process(COMMAND ${TINT} --format ${f} --validate ${module}
                  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "tint --format ${f} failed (${rc}):\n${err}")
  endif()
  file(WRITE ${OUT}/module.${f}.txt "${out}")
  string(LENGTH "${out}" n)
  message(STATUS "${f}: compiled (${n} bytes)")
endforeach()
