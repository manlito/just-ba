// Python bindings, first version (document 03: thin). The C++ types are exposed as
// they are; arrays cross as NumPy arrays (copied in and out in this version).
// The Python layer in just_ba/__init__.py adds conventions and conveniences.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "jba/camera_model.h"
#include "jba/covariance.h"
#include "jba/export.h"
#include "jba/initialize.h"
#include "jba/priors.h"
#include "jba/problem.h"
#include "jba/problem_io.h"
#include "jba/refine.h"
#include "jba/report.h"
#include "jba/solver.h"
#include "jba/tracks.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// ---------------------------------------------------------------------------
// Arrays.

template <typename T>
using In = nb::ndarray<const T, nb::c_contig, nb::device::cpu>;
template <typename T>
using Out = nb::ndarray<nb::numpy, T>;

template <typename T>
Out<T> to_numpy(const std::vector<T>& v, std::vector<std::size_t> tail = {}) {
  std::size_t per = 1;
  for (auto d : tail) per *= d;
  std::vector<std::size_t> shape = {per == 0 ? 0 : v.size() / per};
  shape.insert(shape.end(), tail.begin(), tail.end());
  T* data = new T[std::max<std::size_t>(v.size(), 1)];
  std::copy(v.begin(), v.end(), data);
  nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
  return Out<T>(data, shape.size(), shape.data(), owner);
}

// Copies an array of shape [n] (tail empty) or [n, tail...] into a vector.
template <typename T>
std::vector<T> from_numpy(const In<T>& a, const std::vector<std::size_t>& tail, const char* name) {
  bool ok = a.ndim() == 1 + tail.size();
  for (std::size_t d = 0; ok && d < tail.size(); ++d) ok = a.shape(1 + d) == tail[d];
  if (!ok) {
    std::string want = "(n";
    for (auto d : tail) want += ", " + std::to_string(d);
    throw nb::value_error((std::string(name) + " must have shape " + want + ")").c_str());
  }
  return std::vector<T>(a.data(), a.data() + a.size());
}

// Anything NumPy can turn into an array of T (lists included).
template <typename T>
In<T> as_array(nb::handle h, const char* name) {
  static const char* dtype = std::is_same_v<T, double>          ? "float64"
                             : std::is_same_v<T, std::int32_t>  ? "int32"
                             : std::is_same_v<T, std::int64_t>  ? "int64"
                                                                : "uint8";
  nb::object arr;
  try {
    arr = nb::module_::import_("numpy").attr("ascontiguousarray")(h, "dtype"_a = dtype);
  } catch (nb::python_error&) {
    throw nb::type_error((std::string(name) + " must be convertible to a " + dtype + " array").c_str());
  }
  return nb::cast<In<T>>(arr);
}

template <typename T>
std::vector<T> from_optional(const std::optional<In<T>>& a, const std::vector<std::size_t>& tail, const char* name) {
  return a ? from_numpy(*a, tail, name) : std::vector<T>{};
}

jba::CameraModel model_of(const nb::handle& h) {
  if (nb::isinstance<nb::str>(h)) {
    const std::string s = nb::cast<std::string>(h);
    if (auto m = jba::model_from_name(s)) return *m;
    throw nb::value_error(("unknown camera model " + s).c_str());
  }
  return nb::cast<jba::CameraModel>(h);
}

jba::LossType loss_type_of(const nb::handle& h) {
  if (nb::isinstance<nb::str>(h)) {
    const std::string s = nb::cast<std::string>(h);
    if (auto t = jba::loss_from_name(s)) return *t;
    throw nb::value_error(("unknown loss " + s + " (none, huber, cauchy, geman_mcclure)").c_str());
  }
  return nb::cast<jba::LossType>(h);
}

// Normalises the quaternion at the start of every row of `width` values. Poses
// from single-precision sources (ARKit and ARCore matrices are float) are a few
// 1e-8 off unit norm, beyond validate()'s 1e-9 (document 12). A zero quaternion is
// left as it is, for validate() to refuse.
void normalise_quaternions(std::vector<double>& v, std::size_t width) {
  for (std::size_t r = 0; r + width <= v.size(); r += width) {
    double* q = &v[r];
    const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (n > 0 && std::isfinite(n))
      for (int a = 0; a < 4; ++a) q[a] /= n;
  }
}

// Normalises rows of three values (directions such as gravity, from float sensors).
void normalise_vectors(std::vector<double>& v) {
  for (std::size_t r = 0; r + 3 <= v.size(); r += 3) {
    double* d = &v[r];
    const double n = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    if (n > 0 && std::isfinite(n))
      for (int a = 0; a < 3; ++a) d[a] /= n;
  }
}

enum class Normalise { None, Quaternion, Vector };

// Problem array fields with a fixed row width. `normalise`: rows start with a
// quaternion, or are 3-vectors, normalised on the way in.
template <typename T>
void array_field(nb::class_<jba::Problem>& c, const char* name, std::vector<T> jba::Problem::*field,
                 std::vector<std::size_t> tail, const char* doc, bool quaternion_rows = false,
                 Normalise normalise = Normalise::None) {
  if (quaternion_rows) normalise = Normalise::Quaternion;
  c.def_prop_rw(
      name, [field, tail](const jba::Problem& p) { return to_numpy(p.*field, tail); },
      [field, tail, name, normalise](jba::Problem& p, nb::handle h) {
        p.*field = from_numpy(as_array<T>(h, name), tail, name);
        if constexpr (std::is_same_v<T, double>) {
          if (normalise == Normalise::Quaternion) normalise_quaternions(p.*field, tail.at(0));
          if (normalise == Normalise::Vector) normalise_vectors(p.*field);
        }
      },
      nb::for_getter(nb::rv_policy::move), doc);
}

// Python callables as iteration callbacks: run with the GIL held; an exception
// stops the solve and is raised again afterwards.
struct Callback {
  nb::object fn;
  std::optional<nb::python_error> error;
  bool operator()(const jba::IterationSummary& it) {
    nb::gil_scoped_acquire gil;
    try {
      nb::object r = fn(it);
      return r.is_none() ? true : nb::cast<bool>(r);
    } catch (nb::python_error& e) {
      error.emplace(std::move(e));
      return false;
    }
  }
};

nb::dict cost_dict(const std::array<double, jba::kNumLossTypes>& c) {
  nb::dict d;
  for (std::size_t t = 0; t < jba::kNumLossTypes; ++t) d[std::string(jba::residual_type_name(t)).c_str()] = c[t];
  return d;
}

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "just-ba: minimal, fast bundle adjustment (compiled core)";

  // --- Enums ---------------------------------------------------------------
  nb::enum_<jba::CameraModel>(m, "CameraModel", "Camera models; names and parameter order follow COLMAP.")
      .value("SIMPLE_PINHOLE", jba::CameraModel::SimplePinhole)
      .value("PINHOLE", jba::CameraModel::Pinhole)
      .value("SIMPLE_RADIAL", jba::CameraModel::SimpleRadial)
      .value("RADIAL", jba::CameraModel::Radial)
      .value("OPENCV", jba::CameraModel::OpenCV)
      .value("OPENCV_FISHEYE", jba::CameraModel::OpenCVFisheye);
  m.def("num_params", [](nb::handle model) { return jba::num_params(model_of(model)); }, "model"_a,
        "Number of intrinsic parameters of a camera model.");

  nb::enum_<jba::LossType>(m, "LossType")
      .value("NONE", jba::LossType::None)
      .value("HUBER", jba::LossType::Huber)
      .value("CAUCHY", jba::LossType::Cauchy)
      .value("GEMAN_MCCLURE", jba::LossType::GemanMcClure);
  nb::enum_<jba::FrameRelation>(m, "FrameRelation", "How a global prior set relates to the working frame (D19).")
      .value("IDENTITY", jba::FrameRelation::Identity)
      .value("RIGID", jba::FrameRelation::Rigid)
      .value("SIMILARITY", jba::FrameRelation::Similarity);
  nb::enum_<jba::DepthKind>(m, "DepthKind", "What a depth reading measures.")
      .value("Z", jba::DepthKind::Z)
      .value("RANGE", jba::DepthKind::Range);
  nb::enum_<jba::DepthForm>(m, "DepthForm", "Form of the depth residual (D87).")
      .value("LINEAR", jba::DepthForm::Linear)
      .value("INVERSE", jba::DepthForm::Inverse);
  m.attr("DEPTH_SCALE") = jba::kDepthScale;
  m.attr("DEPTH_OFFSET") = jba::kDepthOffset;
  nb::enum_<jba::GaugeMode>(m, "GaugeMode").value("AUTO", jba::GaugeMode::Auto).value("NONE", jba::GaugeMode::None);
  nb::enum_<jba::BackendKind>(m, "Backend",
                              "Where a solve runs: CPU (double, the reference), Metal (Apple GPUs) or CUDA "
                              "(NVIDIA GPUs).")
      .value("CPU", jba::BackendKind::Cpu)
      .value("METAL", jba::BackendKind::Metal)
      .value("CUDA", jba::BackendKind::Cuda)
      .value("WEBGPU", jba::BackendKind::WebGpu);
  nb::enum_<jba::LinearSolverType>(m, "LinearSolver",
                                   "Inner solve: AUTO (the backend's fastest), ITERATIVE_SCHUR (every backend) "
                                   "or SPARSE_SCHUR (Metal; CUDA with cuDSS).")
      .value("AUTO", jba::LinearSolverType::Auto)
      .value("ITERATIVE_SCHUR", jba::LinearSolverType::IterativeSchur)
      .value("SPARSE_SCHUR", jba::LinearSolverType::SparseSchur);
  m.def("backend_available", &jba::backend_available, "kind"_a,
        "Whether this build has the backend and the machine can run it.");
  m.def("resolve_linear_solver", &jba::resolve_linear_solver, "type"_a, "backend"_a,
        "The solver AUTO stands for on a backend before the problem is seen; other types as they are.");
  nb::enum_<jba::Termination>(m, "Termination")
      .value("CONVERGENCE", jba::Termination::Convergence)
      .value("NO_CONVERGENCE", jba::Termination::NoConvergence)
      .value("FAILURE", jba::Termination::Failure)
      .value("CANCELLED", jba::Termination::Cancelled);
  nb::enum_<jba::ConflictPolicy>(m, "ConflictPolicy")
      .value("DROP", jba::ConflictPolicy::Drop)
      .value("REPAIR", jba::ConflictPolicy::Repair);

  // --- Problem -------------------------------------------------------------
  nb::class_<jba::Problem> problem(
      m, "Problem",
      "A bundle adjustment problem (document 11). Poses map world to camera, stored as qw qx qy qz tx ty tz; "
      "camera axes x right, y down, z forward. Array attributes are copied in and out.");
  problem.def(nb::init<>())
      .def("__copy__", [](const jba::Problem& p) { return jba::Problem(p); })
      .def("copy", [](const jba::Problem& p) { return jba::Problem(p); }, "A deep copy.")
      .def_prop_rw(
          "model", [](const jba::Problem& p) { return p.model; },
          [](jba::Problem& p, nb::handle h) { p.model = model_of(h); }, "Camera model (CameraModel or its name).")
      .def_prop_rw(
          "intrinsics",
          [](const jba::Problem& p) {
            return to_numpy(p.intrinsics, {static_cast<std::size_t>(jba::num_params(p.model))});
          },
          [](jba::Problem& p, nb::handle h) {
            const auto a = as_array<double>(h, "intrinsics");
            const auto np = static_cast<std::size_t>(jba::num_params(p.model));
            if (a.ndim() == 1 && a.shape(0) == np) p.intrinsics.assign(a.data(), a.data() + np);
            else p.intrinsics = from_numpy(a, {np}, "intrinsics");
          },
          nb::for_getter(nb::rv_policy::move),
          "Intrinsics blocks, (B, num_params(model)); one block (a 1-d array) is the normal case. Set the model first.")
      .def_prop_ro("num_images", &jba::Problem::num_images)
      .def_prop_ro("num_points", &jba::Problem::num_points)
      .def_prop_ro("num_observations", &jba::Problem::num_observations)
      .def_prop_ro("num_relative_priors", &jba::Problem::num_relative_priors)
      .def_prop_ro("num_global_priors", &jba::Problem::num_global_priors)
      .def_prop_ro("num_gravity_priors", &jba::Problem::num_gravity_priors)
      .def_prop_ro("num_lines", &jba::Problem::num_lines)
      .def_prop_ro("num_depth_observations", &jba::Problem::num_depth_observations)
      .def("has_manhattan", &jba::Problem::has_manhattan, "Whether solve() includes the Manhattan term.")
      .def_prop_ro("num_prior_sets", &jba::Problem::num_prior_sets)
      .def("validate", [](const jba::Problem& p) { return jba::validate(p); },
           "Empty string when valid, otherwise the first problem found.");
  array_field(problem, "intrinsics_free", &jba::Problem::intrinsics_free, {},
              "Per intrinsics block: bit mask of freed groups (1 focal, 2 principal point, 4 distortion).");
  array_field(problem, "poses", &jba::Problem::poses, {7},
              "(N, 7) world-to-camera poses; quaternions are normalised on the way in.", true);
  array_field(problem, "image_intrinsics", &jba::Problem::image_intrinsics, {}, "(N,) intrinsics block per image.");
  array_field(problem, "image_fixed", &jba::Problem::image_fixed, {}, "(N,) 1 holds a pose fixed.");
  array_field(problem, "points", &jba::Problem::points, {3}, "(M, 3) world points.");
  array_field(problem, "point_fixed", &jba::Problem::point_fixed, {}, "(M,) 1 holds a point fixed.");
  array_field(problem, "obs_image", &jba::Problem::obs_image, {}, "(K,) image of each observation.");
  array_field(problem, "obs_point", &jba::Problem::obs_point, {}, "(K,) point of each observation.");
  array_field(problem, "obs_xy", &jba::Problem::obs_xy, {2}, "(K, 2) keypoints in pixels.");
  array_field(problem, "obs_sigma", &jba::Problem::obs_sigma, {}, "(K,) keypoint standard deviations, pixels.");
  array_field(problem, "rel_images", &jba::Problem::rel_images, {2}, "(R, 2) images i, j of each relative prior.");
  array_field(problem, "rel_pose", &jba::Problem::rel_pose, {7},
              "(R, 7) relative priors: the motion T_j T_i^-1 (camera i to camera j coordinates).", true);
  array_field(problem, "rel_sigma", &jba::Problem::rel_sigma, {2},
              "(R, 2) rotation (rad) and translation (m) standard deviations; inf leaves a part out.");
  array_field(problem, "rel_sqrt_info", &jba::Problem::rel_sqrt_info, {6, 6}, "(R, 6, 6) whitening matrices.");
  array_field(problem, "glob_image", &jba::Problem::glob_image, {}, "(G,) image of each global prior.");
  array_field(problem, "glob_set", &jba::Problem::glob_set, {}, "(G,) prior set of each global prior.");
  array_field(problem, "glob_pose", &jba::Problem::glob_pose, {7}, "(G, 7) world-to-camera poses in the set's frame.",
              true);
  array_field(problem, "glob_sigma", &jba::Problem::glob_sigma, {2},
              "(G, 2) rotation (rad) and position (set units) standard deviations.");
  array_field(problem, "glob_sqrt_info", &jba::Problem::glob_sqrt_info, {6, 6}, "(G, 6, 6) whitening matrices.");
  array_field(problem, "set_transform", &jba::Problem::set_transform, {8},
              "(S, 8) qw qx qy qz tx ty tz s per prior set: X_working = s R X_set + t.", true);
  array_field(problem, "gravity_world", &jba::Problem::gravity_world, {},
              "(3,) direction of gravity in the working frame; empty means (0, -1, 0), ARKit's and ARCore's world. "
              "Normalised on the way in.", false, Normalise::Vector);
  array_field(problem, "grav_image", &jba::Problem::grav_image, {}, "(Q,) image of each gravity prior.");
  array_field(problem, "grav_dir", &jba::Problem::grav_dir, {3},
              "(Q, 3) direction of gravity measured in each image's camera axes (x right, y down, z forward); "
              "normalised on the way in.", false, Normalise::Vector);
  array_field(problem, "grav_sigma", &jba::Problem::grav_sigma, {}, "(Q,) gravity prior standard deviations, rad.");
  array_field(problem, "obs_depth", &jba::Problem::obs_depth, {},
              "(K,) depth reading of each observation, NaN for none; empty: no depth.");
  array_field(problem, "obs_depth_sigma", &jba::Problem::obs_depth_sigma, {}, "(K,) depth reading sigmas.");
  problem.def_rw("depth_kind", &jba::Problem::depth_kind).def_rw("depth_form", &jba::Problem::depth_form);
  array_field(problem, "depth_correction", &jba::Problem::depth_correction, {2},
              "(N, 2) per image (a, b): a d + b is the corrected reading; empty: (1, 0).");
  array_field(problem, "depth_correction_free", &jba::Problem::depth_correction_free, {},
              "(N,) bits of the correction that are estimated: DEPTH_SCALE, DEPTH_OFFSET.");
  array_field(problem, "line_image", &jba::Problem::line_image, {}, "(L,) image of each line segment.");
  array_field(problem, "line_xy", &jba::Problem::line_xy, {4}, "(L, 4) segment endpoints x0 y0 x1 y1, pixels.");
  array_field(problem, "line_sigma", &jba::Problem::line_sigma, {}, "(L,) endpoint standard deviations, pixels.");
  array_field(problem, "line_axis", &jba::Problem::line_axis, {},
              "(L,) building axis of each segment: 0, 1 horizontal, 2 vertical, -1 none. Set by assign_lines.");
  array_field(problem, "manhattan_frame", &jba::Problem::manhattan_frame, {},
              "(4,) quaternion from the building's axes to the working frame; empty: no Manhattan term.", false,
              Normalise::Quaternion);
  problem.def_rw("manhattan_dof", &jba::Problem::manhattan_dof,
                 "1: the frame's heading about the vertical only; 3: its whole rotation.");
  array_field(problem, "loss_type", &jba::Problem::loss_type, {},
              "Stored losses (loss.* sections): LossType values for reprojection, relative prior, global prior, "
              "gravity prior. "
              "Empty: the solver options decide. Applied with with_problem_losses.");
  array_field(problem, "loss_scale", &jba::Problem::loss_scale, {}, "Scale of each stored loss.");
  problem.def_prop_rw(
      "set_relation",
      [](const jba::Problem& p) {
        std::vector<std::int32_t> v(p.set_relation.size());
        for (std::size_t a = 0; a < v.size(); ++a) v[a] = static_cast<std::int32_t>(p.set_relation[a]);
        return to_numpy(v);
      },
      [](jba::Problem& p, nb::handle h) {
        p.set_relation.clear();
        for (nb::handle item : h) {
          if (nb::isinstance<jba::FrameRelation>(item)) p.set_relation.push_back(nb::cast<jba::FrameRelation>(item));
          else p.set_relation.push_back(static_cast<jba::FrameRelation>(nb::cast<std::int32_t>(item)));
        }
      },
      nb::for_getter(nb::rv_policy::move),
      "(S,) relation of each prior set: FrameRelation values (0 identity, 1 rigid, 2 similarity).");

  m.def("read_problem", &jba::read_problem, "path"_a, "Reads a .jba problem file.");
  m.def("write_problem", &jba::write_problem, "path"_a, "problem"_a, "Writes a .jba problem file.");
  m.def("read_bal", &jba::read_bal, "path"_a, "Reads a Bundle Adjustment in the Large problem.");
  m.def(
      "add_relative_priors_from_poses",
      [](jba::Problem& p, nb::handle poses, nb::handle pairs) {
        auto pv = from_numpy(as_array<double>(poses, "poses"), {7}, "poses");
        normalise_quaternions(pv, 7);
        jba::add_relative_priors_from_poses(p, pv, from_numpy(as_array<std::int32_t>(pairs, "pairs"), {2}, "pairs"));
      },
      "problem"_a, "poses"_a, "pairs"_a,
      "Appends relative priors T_j T_i^-1 computed from per-image poses (N, 7) for each pair (i, j) in pairs (P, 2).");

  // --- Options -------------------------------------------------------------
  nb::class_<jba::Loss>(m, "Loss", "Robust loss, Ceres's definitions; scale in standard deviations.")
      .def(nb::init<>())
      .def("__init__", [](jba::Loss* l, nb::handle type, double scale) { new (l) jba::Loss{loss_type_of(type), scale}; },
           "type"_a, "scale"_a = 1.0)
      .def_rw("type", &jba::Loss::type)
      .def_rw("scale", &jba::Loss::scale)
      .def("__repr__", [](const jba::Loss& l) {
        return "Loss('" + std::string(jba::loss_name(l.type)) + "', " + std::to_string(l.scale) + ")";
      });

  nb::class_<jba::PriorDefaults>(m, "PriorDefaults", "Weights for priors given without any (provisional, D67).")
      .def(nb::init<>())
      .def_rw("relative_rotation_sigma", &jba::PriorDefaults::relative_rotation_sigma)
      .def_rw("relative_rotation_per_radian", &jba::PriorDefaults::relative_rotation_per_radian)
      .def_rw("relative_translation_sigma", &jba::PriorDefaults::relative_translation_sigma)
      .def_rw("relative_translation_per_metre", &jba::PriorDefaults::relative_translation_per_metre)
      .def_rw("global_rotation_sigma", &jba::PriorDefaults::global_rotation_sigma)
      .def_rw("global_position_sigma", &jba::PriorDefaults::global_position_sigma)
      .def_rw("gravity_sigma", &jba::PriorDefaults::gravity_sigma)
      .def_rw("line_endpoint_sigma", &jba::PriorDefaults::line_endpoint_sigma)
      .def_rw("line_min_sigma", &jba::PriorDefaults::line_min_sigma)
      .def_rw("depth_sigma", &jba::PriorDefaults::depth_sigma)
      .def_rw("depth_sigma_per_metre", &jba::PriorDefaults::depth_sigma_per_metre)
      .def_rw("depth_scale_sigma", &jba::PriorDefaults::depth_scale_sigma)
      .def_rw("depth_offset_sigma", &jba::PriorDefaults::depth_offset_sigma);

  nb::class_<jba::SolverOptions>(m, "SolverOptions", "Levenberg-Marquardt options; defaults are Ceres's.")
      .def(nb::init<>())
      .def_rw("max_iterations", &jba::SolverOptions::max_iterations)
      .def_rw("function_tolerance", &jba::SolverOptions::function_tolerance)
      .def_rw("gradient_tolerance", &jba::SolverOptions::gradient_tolerance)
      .def_rw("parameter_tolerance", &jba::SolverOptions::parameter_tolerance)
      .def_rw("initial_trust_region_radius", &jba::SolverOptions::initial_trust_region_radius)
      .def_rw("max_trust_region_radius", &jba::SolverOptions::max_trust_region_radius)
      .def_rw("min_trust_region_radius", &jba::SolverOptions::min_trust_region_radius)
      .def_rw("min_relative_decrease", &jba::SolverOptions::min_relative_decrease)
      .def_rw("max_consecutive_invalid_steps", &jba::SolverOptions::max_consecutive_invalid_steps)
      .def_rw("max_linear_iterations", &jba::SolverOptions::max_linear_iterations)
      .def_rw("min_linear_iterations", &jba::SolverOptions::min_linear_iterations)
      .def_rw("eta", &jba::SolverOptions::eta)
      .def_rw("loss", &jba::SolverOptions::loss, "Loss on reprojection residuals.")
      .def_rw("relative_prior_loss", &jba::SolverOptions::relative_prior_loss)
      .def_rw("global_prior_loss", &jba::SolverOptions::global_prior_loss)
      .def_rw("gravity_loss", &jba::SolverOptions::gravity_loss)
      .def_rw("line_loss", &jba::SolverOptions::line_loss)
      .def_rw("depth_loss", &jba::SolverOptions::depth_loss)
      .def_rw("prior_defaults", &jba::SolverOptions::prior_defaults)
      .def_rw("gauge", &jba::SolverOptions::gauge)
      .def_rw("backend", &jba::SolverOptions::backend)
      .def_rw("linear_solver", &jba::SolverOptions::linear_solver)
      .def_rw("num_threads", &jba::SolverOptions::num_threads, "0: all hardware threads. Results do not depend on it.");

  // --- Solve ---------------------------------------------------------------
  nb::class_<jba::IterationSummary>(m, "IterationSummary")
      .def_ro("iteration", &jba::IterationSummary::iteration)
      .def_ro("cost", &jba::IterationSummary::cost)
      .def_ro("cost_change", &jba::IterationSummary::cost_change)
      .def_ro("gradient_max_norm", &jba::IterationSummary::gradient_max_norm)
      .def_ro("step_norm", &jba::IterationSummary::step_norm)
      .def_ro("relative_decrease", &jba::IterationSummary::relative_decrease)
      .def_ro("trust_region_radius", &jba::IterationSummary::trust_region_radius)
      .def_ro("linear_iterations", &jba::IterationSummary::linear_iterations)
      .def_ro("step_valid", &jba::IterationSummary::step_valid)
      .def_ro("step_accepted", &jba::IterationSummary::step_accepted)
      .def_ro("time_seconds", &jba::IterationSummary::time_seconds);
  nb::class_<jba::Gauge>(m, "Gauge")
      .def_ro("fixed_images", &jba::Gauge::fixed_images)
      .def_ro("heading_images", &jba::Gauge::heading_images,
              "Poses whose translation and heading are held, their tilt free (gravity known, D83).")
      .def_ro("translation_images", &jba::Gauge::translation_images,
              "Poses whose translation alone is held; the Manhattan frame sets their rotation (D86).")
      .def_ro("manhattan_frame_held", &jba::Gauge::manhattan_frame_held)
      .def_ro("fixed_translations", &jba::Gauge::fixed_translations)
      .def_ro("fixed_set_scales", &jba::Gauge::fixed_set_scales);
  nb::class_<jba::SolverSummary>(m, "SolverSummary")
      .def_ro("termination", &jba::SolverSummary::termination)
      .def_ro("message", &jba::SolverSummary::message)
      .def_ro("initial_cost", &jba::SolverSummary::initial_cost)
      .def_ro("final_cost", &jba::SolverSummary::final_cost)
      .def_prop_ro("initial_cost_by_type", [](const jba::SolverSummary& s) { return cost_dict(s.initial_cost_by_type); },
                   "Initial cost by residual type, a dict.")
      .def_prop_ro("final_cost_by_type", [](const jba::SolverSummary& s) { return cost_dict(s.final_cost_by_type); },
                   "Final cost by residual type, a dict.")
      .def_ro("iterations", &jba::SolverSummary::iterations)
      .def_ro("accepted_steps", &jba::SolverSummary::accepted_steps)
      .def_ro("linear_iterations", &jba::SolverSummary::linear_iterations)
      .def_ro("time_seconds", &jba::SolverSummary::time_seconds, "Wall time of the whole call, set-up included.")
      .def_ro("setup_seconds", &jba::SolverSummary::setup_seconds, "The part of it before the iterations.")
      .def_ro("linear_solver", &jba::SolverSummary::linear_solver, "The linear solver the steps used.")
      .def_ro("linear_solver_fallbacks", &jba::SolverSummary::linear_solver_fallbacks,
              "sparse_schur steps solved by conjugate gradients instead.")
      .def_ro("history", &jba::SolverSummary::history)
      .def_ro("gauge", &jba::SolverSummary::gauge)
      .def("__repr__", [](const jba::SolverSummary& s) {
        return "SolverSummary(" + std::string(jba::termination_name(s.termination)) + ", cost " +
               std::to_string(s.initial_cost) + " -> " + std::to_string(s.final_cost) + ", " +
               std::to_string(s.iterations) + " iterations)";
      });

  m.def(
      "solve",
      [](jba::Problem& p, const jba::SolverOptions& o, nb::object callback) {
        jba::SolverSummary s;
        if (callback.is_none()) {
          nb::gil_scoped_release release;
          s = jba::solve(p, o);
        } else {
          Callback cb{callback, std::nullopt};
          {
            nb::gil_scoped_release release;
            s = jba::solve(p, o, [&cb](const jba::IterationSummary& it) { return cb(it); });
          }
          if (cb.error) throw std::move(*cb.error);
        }
        return s;
      },
      "problem"_a, "options"_a = jba::SolverOptions{}, "callback"_a = nb::none(),
      "Solves in place on the CPU. callback(IterationSummary) is called after every iteration; "
      "returning False stops the solve.");
  m.def("with_problem_losses", &jba::with_problem_losses, "options"_a, "problem"_a,
        "A copy of the options with the problem's stored losses (loss_type, loss_scale) in place of theirs.");
  m.def("store_losses", &jba::store_losses, "problem"_a, "options"_a,
        "Stores the options' losses in the problem, one per residual type, written as the loss.* sections.");
  m.def("choose_gauge", &jba::choose_gauge, "problem"_a, "The gauge GaugeMode.AUTO would hold fixed.");

  nb::class_<jba::ResidualReport>(m, "ResidualReport", "Per-residual diagnostics at the current parameters.")
      .def_ro("reprojection_cost", &jba::ResidualReport::reprojection_cost)
      .def_ro("relative_prior_cost", &jba::ResidualReport::relative_prior_cost)
      .def_ro("global_prior_cost", &jba::ResidualReport::global_prior_cost)
      .def_ro("gravity_cost", &jba::ResidualReport::gravity_cost)
      .def_ro("line_cost", &jba::ResidualReport::line_cost)
      .def_ro("depth_cost", &jba::ResidualReport::depth_cost)
      .def_prop_ro("depth_refined", [](const jba::ResidualReport& r) { return to_numpy(r.depth_refined); },
                   "Per observation: refined depth of its point, NaN without a reading.", nb::rv_policy::move)
      .def_prop_ro("depth_corrected", [](const jba::ResidualReport& r) { return to_numpy(r.depth_corrected); },
                   "Per observation: the corrected reading a d + b.", nb::rv_policy::move)
      .def_prop_ro("depth_weight", [](const jba::ResidualReport& r) { return to_numpy(r.depth_weight); }, nb::rv_policy::move)
      .def_prop_ro("line_angle", [](const jba::ResidualReport& r) { return to_numpy(r.line_angle); },
                   "Per segment: angle by which its axis misses its plane, rad; NaN when not in the term.",
                   nb::rv_policy::move)
      .def_prop_ro("line_weight", [](const jba::ResidualReport& r) { return to_numpy(r.line_weight); }, nb::rv_policy::move)
      .def_prop_ro("reprojection_error", [](const jba::ResidualReport& r) { return to_numpy(r.reprojection_error); }, nb::rv_policy::move)
      .def_prop_ro("in_front", [](const jba::ResidualReport& r) {
        std::vector<std::uint8_t> v(r.in_front.begin(), r.in_front.end());
        return to_numpy(v);
      }, nb::rv_policy::move)
      .def_prop_ro("relative_prior_norm", [](const jba::ResidualReport& r) { return to_numpy(r.relative_prior_norm); }, nb::rv_policy::move)
      .def_prop_ro("relative_prior_weight", [](const jba::ResidualReport& r) { return to_numpy(r.relative_prior_weight); }, nb::rv_policy::move)
      .def_prop_ro("global_prior_norm", [](const jba::ResidualReport& r) { return to_numpy(r.global_prior_norm); }, nb::rv_policy::move)
      .def_prop_ro("global_prior_weight", [](const jba::ResidualReport& r) { return to_numpy(r.global_prior_weight); }, nb::rv_policy::move)
      .def_prop_ro("downweighted_relative", [](const jba::ResidualReport& r) { return to_numpy(r.downweighted_relative); }, nb::rv_policy::move)
      .def_prop_ro("downweighted_global", [](const jba::ResidualReport& r) { return to_numpy(r.downweighted_global); }, nb::rv_policy::move)
      .def_prop_ro("gravity_angle", [](const jba::ResidualReport& r) { return to_numpy(r.gravity_angle); },
                   "Per gravity prior: angle between measured and solved gravity, rad.", nb::rv_policy::move)
      .def_prop_ro("gravity_weight", [](const jba::ResidualReport& r) { return to_numpy(r.gravity_weight); }, nb::rv_policy::move)
      .def_prop_ro("downweighted_gravity", [](const jba::ResidualReport& r) { return to_numpy(r.downweighted_gravity); }, nb::rv_policy::move);
  m.def("evaluate", &jba::evaluate, "problem"_a, "options"_a = jba::SolverOptions{}, "downweight_threshold"_a = 0.1,
        nb::call_guard<nb::gil_scoped_release>(), "Cost by residual type, reprojection errors, prior weights.");

  // --- Starting poses --------------------------------------------------------
  nb::class_<jba::PoseInitialization>(m, "PoseInitialization")
      .def_prop_ro("poses", [](const jba::PoseInitialization& i) { return to_numpy(i.poses, {7}); }, nb::rv_policy::move)
      .def_prop_ro("set_transform", [](const jba::PoseInitialization& i) { return to_numpy(i.set_transform, {8}); }, nb::rv_policy::move)
      .def_prop_ro("group", [](const jba::PoseInitialization& i) { return to_numpy(i.group); }, nb::rv_policy::move)
      .def_ro("num_groups", &jba::PoseInitialization::num_groups);
  m.def("initialize_poses", &jba::initialize_poses, "problem"_a, "defaults"_a = jba::PriorDefaults{},
        "Starting poses from the priors alone, and the groups of images they connect.");

  // --- Tracks --------------------------------------------------------------
  nb::class_<jba::TriangulationOptions>(m, "TriangulationOptions")
      .def(nb::init<>())
      .def_rw("max_angle_error_deg", &jba::TriangulationOptions::max_angle_error_deg)
      .def_rw("min_tri_angle_deg", &jba::TriangulationOptions::min_tri_angle_deg)
      .def_rw("min_views", &jba::TriangulationOptions::min_views)
      .def_rw("max_samples", &jba::TriangulationOptions::max_samples);
  nb::class_<jba::TrackOptions>(m, "TrackOptions")
      .def(nb::init<>())
      .def_rw("min_views", &jba::TrackOptions::min_views)
      .def_rw("max_views", &jba::TrackOptions::max_views)
      .def_rw("conflict_policy", &jba::TrackOptions::conflict_policy)
      .def_rw("check_clean", &jba::TrackOptions::check_clean)
      .def_rw("triangulation", &jba::TrackOptions::triangulation);
  nb::class_<jba::TrackStats>(m, "TrackStats")
      .def_ro("rejected_matches", &jba::TrackStats::rejected_matches)
      .def_ro("components", &jba::TrackStats::components)
      .def_ro("clean", &jba::TrackStats::clean)
      .def_ro("conflicted", &jba::TrackStats::conflicted)
      .def_ro("tracks", &jba::TrackStats::tracks)
      .def_ro("dropped_observations", &jba::TrackStats::dropped_observations);
  nb::class_<jba::Tracks>(m, "Tracks",
                          "Track t has observations begin[t]..begin[t+1]-1, each an (image, keypoint) pair.")
      .def(nb::init<>())
      .def("__init__",
           [](jba::Tracks* t, nb::handle begin, nb::handle image, nb::handle keypoint) {
             new (t) jba::Tracks{from_numpy(as_array<std::int64_t>(begin, "begin"), {}, "begin"),
                                 from_numpy(as_array<std::int32_t>(image, "image"), {}, "image"),
                                 from_numpy(as_array<std::int32_t>(keypoint, "keypoint"), {}, "keypoint")};
             if (t->begin.empty() || t->begin.front() != 0 ||
                 t->begin.back() != static_cast<std::int64_t>(t->image.size()) || t->image.size() != t->keypoint.size())
               throw nb::value_error("tracks: begin must start at 0 and end at len(image) == len(keypoint)");
           },
           "begin"_a, "image"_a, "keypoint"_a)
      .def("__len__", &jba::Tracks::size)
      .def_prop_ro("begin", [](const jba::Tracks& t) { return to_numpy(t.begin); }, nb::rv_policy::move)
      .def_prop_ro("image", [](const jba::Tracks& t) { return to_numpy(t.image); }, nb::rv_policy::move)
      .def_prop_ro("keypoint", [](const jba::Tracks& t) { return to_numpy(t.keypoint); }, nb::rv_policy::move);

  // Keypoints as a list of (n_i, 2) arrays, one per image.
  auto keypoints_of = [](nb::handle per_image) {
    jba::Keypoints k;
    k.begin.assign(1, 0);
    for (nb::handle h : per_image) {
      const auto a = as_array<double>(h, "keypoints[i]");
      if (a.size() == 0) {
        k.begin.push_back(k.begin.back());
        continue;
      }
      const auto v = from_numpy(a, {2}, "keypoints[i]");
      k.xy.insert(k.xy.end(), v.begin(), v.end());
      k.begin.push_back(static_cast<std::int64_t>(k.xy.size() / 2));
    }
    return k;
  };

  m.def(
      "build_tracks",
      [keypoints_of](nb::handle keypoints, nb::handle matches, const jba::TrackOptions& o,
                     std::optional<const jba::Problem*> cameras) {
        const auto k = keypoints_of(keypoints);
        const auto mv = from_numpy(as_array<std::int32_t>(matches, "matches"), {4}, "matches");
        jba::TrackStats st;
        jba::Tracks t;
        if (cameras && *cameras) {
          const jba::Problem& p = **cameras;
          jba::Cameras c{p.model, p.intrinsics, p.image_intrinsics, p.poses};
          nb::gil_scoped_release release;
          t = jba::build_tracks(k, mv, o, &c, &st);
        } else {
          nb::gil_scoped_release release;
          t = jba::build_tracks(k, mv, o, nullptr, &st);
        }
        return std::make_pair(t, st);
      },
      "keypoints"_a, "matches"_a, "options"_a = jba::TrackOptions{}, "cameras"_a = nb::none(),
      "Tracks from pairwise matches (M, 4): image_a, keypoint_a, image_b, keypoint_b. With `cameras` (a Problem "
      "giving the model, intrinsics and poses) conflicted components are repaired geometrically. Returns "
      "(Tracks, TrackStats).");

  // --- Refine --------------------------------------------------------------
  nb::class_<jba::Covariance>(m, "Covariance", "Uncertainty of a solved problem (document 02).")
      .def_prop_ro("pose", [](const jba::Covariance& c) { return to_numpy(c.pose, {6, 6}); },
                   "(N, 6, 6) per image, pose tangent: rotation (rad, camera axes), then translation.",
                   nb::rv_policy::move)
      .def_prop_ro("centre", [](const jba::Covariance& c) { return to_numpy(c.centre, {3, 3}); },
                   "(N, 3, 3) per image, camera centre in the working frame.", nb::rv_policy::move)
      .def_prop_ro("points", [](const jba::Covariance& c) { return to_numpy(c.points); }, nb::rv_policy::move)
      .def_prop_ro("point", [](const jba::Covariance& c) { return to_numpy(c.point, {3, 3}); },
                   "(P, 3, 3) per requested point.", nb::rv_policy::move)
      .def_ro("gauge", &jba::Covariance::gauge);
  m.def(
      "covariance",
      [](const jba::Problem& p, const jba::SolverOptions& o, nb::handle points, bool all_points, std::size_t max_unknowns) {
        jba::CovarianceOptions co;
        if (!points.is_none()) co.points = from_numpy(as_array<std::int32_t>(points, "points"), {}, "points");
        co.all_points = all_points;
        co.max_unknowns = max_unknowns;
        nb::gil_scoped_release release;
        return jba::covariance(p, o, co);
      },
      "problem"_a, "options"_a = jba::SolverOptions{}, "points"_a = nb::none(), "all_points"_a = false,
      "max_unknowns"_a = 20000,
      "Covariances of the poses (and of the given points) of a solved problem, relative to the gauge.");

  nb::class_<jba::ManhattanOptions>(m, "ManhattanOptions", "Line assignment and the support test (document 02, section 5).")
      .def(nb::init<>())
      .def_rw("dof", &jba::ManhattanOptions::dof)
      .def_rw("max_angle_deg", &jba::ManhattanOptions::max_angle_deg)
      .def_rw("min_support", &jba::ManhattanOptions::min_support)
      .def_rw("min_segments", &jba::ManhattanOptions::min_segments)
      .def_rw("horizontal_plane_deg", &jba::ManhattanOptions::horizontal_plane_deg);
  nb::class_<jba::ManhattanAssignment>(m, "ManhattanAssignment")
      .def_ro("accepted", &jba::ManhattanAssignment::accepted)
      .def_ro("message", &jba::ManhattanAssignment::message)
      .def_ro("support", &jba::ManhattanAssignment::support)
      .def_ro("unusable", &jba::ManhattanAssignment::unusable)
      .def_prop_ro("frame", [](const jba::ManhattanAssignment& a) { return to_numpy(a.frame); }, nb::rv_policy::move)
      .def_prop_ro("axis", [](const jba::ManhattanAssignment& a) { return to_numpy(a.axis); }, nb::rv_policy::move)
      .def_prop_ro("assigned", [](const jba::ManhattanAssignment& a) {
        return std::vector<int>(a.assigned, a.assigned + 3);
      })
      .def("__repr__", [](const jba::ManhattanAssignment& a) {
        return std::string("ManhattanAssignment(") + (a.accepted ? "accepted" : "refused") + ", support " +
               std::to_string(a.support) + ")";
      });
  m.def(
      "assign_lines",
      [](const jba::Problem& p, const jba::ManhattanOptions& o, nb::handle frame) {
        std::vector<double> f;
        if (!frame.is_none()) f = from_numpy(as_array<double>(frame, "frame"), {}, "frame");
        if (!f.empty() && f.size() != 4) throw nb::value_error("frame must hold 4 values");
        return jba::assign_lines(p, o, f.empty() ? nullptr : f.data());
      },
      "problem"_a, "options"_a = jba::ManhattanOptions{}, "frame"_a = nb::none(),
      "Estimates the Manhattan frame from the segments and current poses, assigns segments to axes, and runs "
      "the support test. With `frame`, assigns to that frame's axes.");
  m.def("apply_assignment", &jba::apply_assignment, "problem"_a, "assignment"_a, "dof"_a = 1,
        "Stores an accepted assignment in the problem (line_axis, manhattan_frame); a refused one clears them.");

  nb::class_<jba::RefineOptions>(m, "RefineOptions")
      .def(nb::init<>())
      .def_rw("solver", &jba::RefineOptions::solver)
      .def_rw("tracks", &jba::RefineOptions::tracks)
      .def_rw("rounds", &jba::RefineOptions::rounds)
      .def_rw("prior_solve", &jba::RefineOptions::prior_solve)
      .def_rw("max_reprojection_error", &jba::RefineOptions::max_reprojection_error)
      .def_rw("min_tri_angle_deg", &jba::RefineOptions::min_tri_angle_deg)
      .def_rw("downweight_threshold", &jba::RefineOptions::downweight_threshold)
      .def_rw("largest_group_only", &jba::RefineOptions::largest_group_only)
      .def_rw("use_manhattan", &jba::RefineOptions::use_manhattan,
              "Manhattan prior from scene's line segments; opt-in.")
      .def_rw("manhattan", &jba::RefineOptions::manhattan)
      .def_rw("manhattan_min_angle_deg", &jba::RefineOptions::manhattan_min_angle_deg)
      .def_rw("min_observations", &jba::RefineOptions::min_observations,
              "Inlier observations an image needs to count as refined (default 30).");
  nb::class_<jba::RefineResult>(m, "RefineResult")
      .def_ro("problem", &jba::RefineResult::problem, "The solved problem.")
      .def_prop_ro("obs_keypoint", [](const jba::RefineResult& r) { return to_numpy(r.obs_keypoint); },
                   "Keypoint (in its image) of each observation of `problem`.", nb::rv_policy::move)
      .def_prop_ro("group", [](const jba::RefineResult& r) { return to_numpy(r.group); }, nb::rv_policy::move)
      .def_ro("num_groups", &jba::RefineResult::num_groups)
      .def_prop_ro("refined", [](const jba::RefineResult& r) {
        std::vector<std::uint8_t> v(r.refined.begin(), r.refined.end());
        return to_numpy(v);
      }, "Per image: solved and constrained by at least `min_observations` inlier observations.",
         nb::rv_policy::move)
      .def_prop_ro("num_observations", [](const jba::RefineResult& r) { return to_numpy(r.num_observations); },
                   "Inlier observations of each image in `problem`.", nb::rv_policy::move)
      .def_ro("report", &jba::RefineResult::report)
      .def_ro("summaries", &jba::RefineResult::summaries)
      .def_ro("track_stats", &jba::RefineResult::track_stats)
      .def_ro("rejected_observations", &jba::RefineResult::rejected_observations)
      .def_prop_ro("relative_prior_index", [](const jba::RefineResult& r) { return to_numpy(r.relative_prior_index); }, nb::rv_policy::move)
      .def_prop_ro("global_prior_index", [](const jba::RefineResult& r) { return to_numpy(r.global_prior_index); }, nb::rv_policy::move)
      .def_prop_ro("gravity_prior_index", [](const jba::RefineResult& r) { return to_numpy(r.gravity_prior_index); }, nb::rv_policy::move)
      .def_ro("manhattan", &jba::RefineResult::manhattan, "The last line assignment, and whether it was accepted.");

  m.def(
      "export_colmap",
      [keypoints_of](const jba::Problem& p, const std::string& directory, std::optional<std::vector<std::string>> names,
                     nb::handle image_size, nb::handle images, nb::handle keypoints, nb::handle obs_keypoint,
                     nb::handle point_color) {
        jba::ColmapExportOptions o;
        if (names) o.image_names = *names;
        if (!image_size.is_none())
          o.image_size = from_numpy(as_array<std::int32_t>(image_size, "image_size"), {2}, "image_size");
        if (!images.is_none()) {
          const auto v = from_numpy(as_array<std::uint8_t>(images, "images"), {}, "images");
          o.images.assign(v.begin(), v.end());
        }
        jba::Keypoints kp;
        if (!keypoints.is_none()) {
          kp = keypoints_of(keypoints);
          o.keypoints = &kp;
          if (obs_keypoint.is_none()) throw nb::value_error("export_colmap: keypoints need obs_keypoint");
          o.obs_keypoint = from_numpy(as_array<std::int32_t>(obs_keypoint, "obs_keypoint"), {}, "obs_keypoint");
        }
        if (!point_color.is_none())
          o.point_color = from_numpy(as_array<std::uint8_t>(point_color, "point_color"), {3}, "point_color");
        jba::export_colmap(p, directory, o);
      },
      "problem"_a, "directory"_a, "image_names"_a = nb::none(), "image_size"_a = nb::none(), "images"_a = nb::none(),
      "keypoints"_a = nb::none(), "obs_keypoint"_a = nb::none(), "point_color"_a = nb::none(),
      "Writes COLMAP's text model (cameras.txt, images.txt, points3D.txt) into an existing directory. With "
      "keypoints (one array per image) and obs_keypoint, POINTS2D lists every keypoint.");
  m.def(
      "write_result",
      [](const std::string& path, const jba::Problem& p, std::optional<const jba::Covariance*> cov,
         std::optional<const jba::ResidualReport*> rep, nb::handle obs_keypoint) {
        std::vector<std::int32_t> kp;
        if (!obs_keypoint.is_none()) kp = from_numpy(as_array<std::int32_t>(obs_keypoint, "obs_keypoint"), {}, "obs_keypoint");
        jba::write_result(path, p, cov ? *cov : nullptr, rep ? *rep : nullptr, obs_keypoint.is_none() ? nullptr : &kp);
      },
      "path"_a, "problem"_a, "covariance"_a = nb::none(), "report"_a = nb::none(), "obs_keypoint"_a = nb::none(),
      "Writes the problem file with result.* sections (covariances, report, keypoint of each observation).");

  m.def(
      "refine",
      [keypoints_of](const jba::Problem& scene, nb::handle keypoints, nb::handle matches,
                     std::optional<const jba::Tracks*> tracks, nb::handle keypoint_sigma,
                     const jba::RefineOptions& o, nb::handle keypoint_depth, nb::handle keypoint_depth_sigma) {
        // Per keypoint, flat in image order, or one array per image.
        auto per_keypoint = [](nb::handle h, const char* name) {
          std::vector<double> out;
          if (nb::isinstance<nb::list>(h) || nb::isinstance<nb::tuple>(h)) {
            for (nb::handle item : h) {
              const auto v = from_numpy(as_array<double>(item, name), {}, name);
              out.insert(out.end(), v.begin(), v.end());
            }
          } else {
            out = from_numpy(as_array<double>(h, name), {}, name);
          }
          return out;
        };
        jba::RefineInput in;
        in.scene = scene;
        in.keypoints = keypoints_of(keypoints);
        if (in.keypoints.num_images() != scene.num_images() && scene.num_images() != 0)
          throw nb::value_error("refine: one keypoint array per image of scene.poses is needed");
        if (!matches.is_none()) in.matches = from_numpy(as_array<std::int32_t>(matches, "matches"), {4}, "matches");
        if (tracks && *tracks) in.tracks = **tracks;
        if (in.matches.empty() && in.tracks.size() == 0) throw nb::value_error("refine: give matches or tracks");
        if (!keypoint_sigma.is_none())
          in.keypoint_sigma = from_numpy(as_array<double>(keypoint_sigma, "keypoint_sigma"), {}, "keypoint_sigma");
        if (!keypoint_depth.is_none()) in.keypoint_depth = per_keypoint(keypoint_depth, "keypoint_depth");
        if (!keypoint_depth_sigma.is_none())
          in.keypoint_depth_sigma = per_keypoint(keypoint_depth_sigma, "keypoint_depth_sigma");
        nb::gil_scoped_release release;
        return jba::refine(in, o);
      },
      "scene"_a, "keypoints"_a, "matches"_a = nb::none(), "tracks"_a = nb::none(), "keypoint_sigma"_a = nb::none(),
      "options"_a = jba::RefineOptions{}, "keypoint_depth"_a = nb::none(), "keypoint_depth_sigma"_a = nb::none(),
      "The main use case end to end: starting poses from the priors in `scene`, tracks from the matches, "
      "triangulation, solve, outlier removal, repeated (D71). keypoint_depth: a reading per keypoint (NaN for "
      "none), flat or one array per image; scene.depth_* say what they are.");
}
