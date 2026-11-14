#include "jba/camera_model.h"

#include <array>

namespace jba {
namespace {

struct ModelInfo {
  CameraModel model;
  std::string_view name;
  int num_params;
  bool has_reprojection;
};

constexpr std::array<ModelInfo, 6> kModels = {{
    {CameraModel::SimplePinhole, "SIMPLE_PINHOLE", 3, true},
    {CameraModel::Pinhole, "PINHOLE", 4, true},
    {CameraModel::SimpleRadial, "SIMPLE_RADIAL", 4, true},
    {CameraModel::Radial, "RADIAL", 5, true},
    {CameraModel::OpenCV, "OPENCV", 8, false},
    {CameraModel::OpenCVFisheye, "OPENCV_FISHEYE", 8, false},
}};

const ModelInfo& info(CameraModel model) { return kModels.at(static_cast<std::size_t>(model)); }

}  // namespace

int num_params(CameraModel model) { return info(model).num_params; }

std::string_view model_name(CameraModel model) { return info(model).name; }

bool has_reprojection(CameraModel model) { return info(model).has_reprojection; }

std::optional<CameraModel> model_from_id(std::int32_t id) {
  if (id < 0 || id >= static_cast<std::int32_t>(kModels.size())) return std::nullopt;
  return kModels[static_cast<std::size_t>(id)].model;
}

std::optional<CameraModel> model_from_name(std::string_view name) {
  for (const auto& m : kModels) {
    if (m.name == name) return m.model;
  }
  return std::nullopt;
}

}  // namespace jba
