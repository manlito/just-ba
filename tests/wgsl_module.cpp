// Writes the WebGPU backend's whole WGSL module to a file, for Tint's command-line tool
// (tests/wgsl_compiles.cmake). Needs no adapter.
#include <cstdio>
#include <fstream>

#include "jba/webgpu/backend.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: jba_wgsl_module <out.wgsl>\n");
    return 2;
  }
  std::ofstream out(argv[1], std::ios::binary);
  out << jba::webgpu::wgsl_module();
  return out ? 0 : 1;
}
