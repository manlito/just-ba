#pragma once

#include <string>
#include <vector>

#include "jba/problem.h"

namespace jba::metal {

// Whether a Metal device is present.
bool available();

// Name of the default Metal device, or an empty string.
std::string device_name();

// Single-precision outputs for every observation, row-major per observation:
// residual 2, J_pose 2 x 6, J_point 2 x 3, J_intrinsics 2 x num_params.
struct ReprojectionBatch {
  std::vector<float> residual;
  std::vector<float> J_pose;
  std::vector<float> J_point;
  std::vector<float> J_intrinsics;
};

// Evaluates the reprojection residual and its Jacobians for every observation on
// the GPU. Milestone 0 spike: inputs are converted to float as they are, without
// the re-centring that the real backend will do (document 03, Precision).
// Throws std::runtime_error when Metal is unavailable or a kernel fails.
ReprojectionBatch evaluate_reprojection(const Problem& problem);

}  // namespace jba::metal
