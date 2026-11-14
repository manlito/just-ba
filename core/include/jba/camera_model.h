#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace jba {

// Camera models. Names, ids and parameter order follow COLMAP (document 02).
enum class CameraModel : std::int32_t {
  SimplePinhole = 0,  // f, cx, cy
  Pinhole = 1,        // fx, fy, cx, cy
  SimpleRadial = 2,   // f, cx, cy, k
  Radial = 3,         // f, cx, cy, k1, k2
  OpenCV = 4,         // fx, fy, cx, cy, k1, k2, p1, p2
  OpenCVFisheye = 5,  // fx, fy, cx, cy, k1, k2, k3, k4
};

// Number of intrinsic parameters of a model.
int num_params(CameraModel model);

// COLMAP name, e.g. "SIMPLE_RADIAL".
std::string_view model_name(CameraModel model);

std::optional<CameraModel> model_from_id(std::int32_t id);
std::optional<CameraModel> model_from_name(std::string_view name);

// Whether the reprojection residual has generated code for this model yet.
bool has_reprojection(CameraModel model);

// Groups of intrinsics that can be freed independently (decision D26). Used as a
// bit mask; 0 means fully fixed, the default.
enum IntrinsicsGroup : std::uint8_t {
  kFocal = 1,
  kPrincipalPoint = 2,
  kDistortion = 4,
};

// Group of parameter `index` of a model.
IntrinsicsGroup intrinsics_group(CameraModel model, int index);

}  // namespace jba
