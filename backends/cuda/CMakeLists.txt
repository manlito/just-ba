# CUDA backend (milestone 5): the shared GPU kernels (D96) compiled by nvcc, and its
# Device. Needs only the CUDA runtime, linked statically (document 03).
find_package(CUDAToolkit REQUIRED)
add_library(jba_cuda STATIC src/backend.cu)
target_include_directories(jba_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/kernels
                           ${PROJECT_SOURCE_DIR}/residuals/generated/cuda)
target_link_libraries(jba_cuda PUBLIC jba_core jba_cpu jba_gpu PRIVATE CUDA::cudart_static jba_internal jba_warnings)
# Strategy B (sparse_schur) with cuDSS, optional (D98): a runtime dependency only when
# enabled. Point CMake at it with -Dcudss_DIR=<cuDSS>/lib/cmake/cudss.
if(JBA_WITH_CUDSS)
  find_package(cudss REQUIRED CONFIG)
  target_link_libraries(jba_cuda PRIVATE cudss)
  target_compile_definitions(jba_cuda PRIVATE JBA_HAVE_CUDSS=1)
endif()
# Device code: IEEE division and square root, fused multiply-add as on Metal; no fast math.
target_compile_options(jba_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--prec-div=true --prec-sqrt=true --fmad=true>)
