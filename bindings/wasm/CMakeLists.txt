# WebAssembly module for the browser (exploration): the library with the WebGPU and
# CPU backends, and the benchmark's synthetic generator, as an ES module
# (just_ba.js + just_ba.wasm). Emscripten only:
#
#   emcmake cmake -S . -B build/wasm -G Ninja -DJBA_WITH_WEBGPU=ON -DJBA_BUILD_WASM=ON
#   cmake --build build/wasm
#
# The host side is single-threaded (no SharedArrayBuffer, so no special headers on the
# server); it waits for the GPU through JSPI, which Chrome and Edge ship.

add_executable(just_ba_wasm src/main.cpp ${PROJECT_SOURCE_DIR}/tests/make_problem.cpp)
set_source_files_properties(${PROJECT_SOURCE_DIR}/tests/make_problem.cpp PROPERTIES COMPILE_DEFINITIONS
                                                                                   main=jba_make_problem_main)
target_include_directories(just_ba_wasm PRIVATE ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(just_ba_wasm PRIVATE jba jba_internal)
set_target_properties(just_ba_wasm PROPERTIES OUTPUT_NAME just_ba SUFFIX ".js")
target_link_options(
  just_ba_wasm PRIVATE
  -sJSPI
  "-sJSPI_EXPORTS=jba_solve,jba_backend_available"
  "-sEXPORTED_FUNCTIONS=_jba_solve,_jba_generate,_jba_backend_available"
  "-sEXPORTED_RUNTIME_METHODS=ccall,FS"
  -sMODULARIZE
  -sEXPORT_ES6
  -sEXPORT_NAME=createJustBa
  -sENVIRONMENT=web
  -sALLOW_MEMORY_GROWTH
  -sMAXIMUM_MEMORY=4GB
  -sSTACK_SIZE=2MB)
# The page and the module side by side, ready to serve.
add_custom_command(TARGET just_ba_wasm POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
                   ${CMAKE_CURRENT_SOURCE_DIR}/web/index.html $<TARGET_FILE_DIR:just_ba_wasm>/index.html)
