// The capture file (.jbc): what the mapper needs from a matcher, and optionally a
// reconstruction of the same capture to compare with (COLMAP's, or the truth of a
// generated scene). Written by demo/tools/colmap_capture.py from a COLMAP database and
// model, and by `jba_demo synth --write`. Sections are listed in demo/README.md.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mapper.h"

namespace demo {

// Another reconstruction of the capture. Observations name a keypoint of the capture.
struct Reference {
  std::vector<double> intrinsics;        // blocks as the capture's: its own (refined) values; empty: none
  std::vector<char> registered;          // per image
  std::vector<double> poses;             // 7 per image, world to camera; meaningful where registered
  std::vector<double> points;            // 3 per point
  std::vector<std::int32_t> obs_image, obs_point, obs_keypoint;
  bool empty() const { return registered.empty(); }
};

struct CaptureFile {
  Capture capture;
  std::vector<std::string> names;        // per image, or empty
  std::vector<std::int32_t> image_size;  // per intrinsics block: width, height; or empty
  Reference reference;
};

// Throws std::runtime_error on a damaged file or one whose sections disagree.
CaptureFile read_capture(const std::string& path);
void write_capture(const std::string& path, const CaptureFile& file);

}  // namespace demo
