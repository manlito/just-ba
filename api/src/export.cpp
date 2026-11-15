#include "jba/export.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "jba/problem_io.h"

namespace jba {
namespace {

std::string path_in(const std::string& dir, const char* name) {
  if (dir.empty()) return name;
  const char last = dir.back();
  return (last == '/' || last == '\\') ? dir + name : dir + "/" + name;
}

std::ofstream open_out(const std::string& path) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error(path + ": cannot open for writing");
  return out;
}

std::string fmt(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.17g", v);
  return buf;
}

RawSection section(const std::string& name, const std::vector<double>& v, std::vector<std::uint64_t> shape) {
  RawSection s;
  s.name = name;
  s.dtype = 5;
  s.shape = std::move(shape);
  s.bytes.resize(v.size() * sizeof(double));
  if (!v.empty()) std::memcpy(s.bytes.data(), v.data(), s.bytes.size());
  return s;
}

RawSection section_i32(const std::string& name, const std::vector<std::int32_t>& v) {
  RawSection s;
  s.name = name;
  s.dtype = 2;
  s.shape = {v.size()};
  s.bytes.resize(v.size() * sizeof(std::int32_t));
  if (!v.empty()) std::memcpy(s.bytes.data(), v.data(), s.bytes.size());
  return s;
}

}  // namespace

void export_colmap(const Problem& p, const std::string& dir, const ColmapExportOptions& o) {
  if (auto err = validate(p); !err.empty()) throw std::runtime_error("export_colmap: invalid problem: " + err);
  const std::size_t n_img = p.num_images(), n_pts = p.num_points(), n_obs = p.num_observations();
  const std::size_t n_blk = p.num_intrinsics_blocks();
  const auto np = static_cast<std::size_t>(num_params(p.model));
  if (!o.image_names.empty() && o.image_names.size() != n_img) throw std::runtime_error("export_colmap: one name per image");
  if (!o.image_size.empty() && o.image_size.size() != 2 * n_blk)
    throw std::runtime_error("export_colmap: image_size needs width and height per intrinsics block");
  if (!o.images.empty() && o.images.size() != n_img) throw std::runtime_error("export_colmap: images needs one flag per image");
  if (!o.point_color.empty() && o.point_color.size() != 3 * n_pts)
    throw std::runtime_error("export_colmap: point_color needs 3 values per point");
  if (o.keypoints && (o.obs_keypoint.size() != n_obs || o.keypoints->num_images() != n_img))
    throw std::runtime_error("export_colmap: keypoints need one array per image and obs_keypoint per observation");

  std::vector<char> exported(n_img, 0);
  for (std::size_t i = 0; i < n_img; ++i) exported[i] = o.images.empty() ? 0 : o.images[i];
  if (o.images.empty())
    for (auto i : p.obs_image) exported[static_cast<std::size_t>(i)] = 1;

  // Reprojection error per observation, for the points' ERROR.
  const ResidualReport rep = evaluate(p, SolverOptions{});

  // POINTS2D of each image: (x, y, point) and the index of each observation in it.
  std::vector<std::vector<std::array<double, 3>>> pts2d(n_img);
  std::vector<std::int64_t> idx2d(n_obs, -1);
  if (o.keypoints) {
    for (std::size_t i = 0; i < n_img; ++i) {
      if (!exported[i]) continue;
      const auto b = o.keypoints->begin[i], e = o.keypoints->begin[i + 1];
      for (auto q = b; q < e; ++q)
        pts2d[i].push_back({o.keypoints->xy[2 * static_cast<std::size_t>(q)], o.keypoints->xy[2 * static_cast<std::size_t>(q) + 1], -1});
    }
  }
  for (std::size_t k = 0; k < n_obs; ++k) {
    const auto i = static_cast<std::size_t>(p.obs_image[k]);
    if (!exported[i]) continue;
    const double j = static_cast<double>(p.obs_point[k] + 1);
    if (o.keypoints) {
      const auto q = static_cast<std::size_t>(o.obs_keypoint[k]);
      pts2d[i].at(q)[2] = j;
      idx2d[k] = static_cast<std::int64_t>(q);
    } else {
      idx2d[k] = static_cast<std::int64_t>(pts2d[i].size());
      pts2d[i].push_back({p.obs_xy[2 * k], p.obs_xy[2 * k + 1], j});
    }
  }

  {
    auto out = open_out(path_in(dir, "cameras.txt"));
    out << "# Camera list with one line of data per camera:\n"
           "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        << "# Number of cameras: " << n_blk << "\n";
    for (std::size_t c = 0; c < n_blk; ++c) {
      const double* k = &p.intrinsics[np * c];
      // Principal point: index 1, 2 for the single-focal models, 2, 3 otherwise.
      const bool single = p.model == CameraModel::SimplePinhole || p.model == CameraModel::SimpleRadial ||
                          p.model == CameraModel::Radial;
      const double cx = single ? k[1] : k[2], cy = single ? k[2] : k[3];
      const long w = o.image_size.empty() ? static_cast<long>(std::ceil(2 * cx)) : o.image_size[2 * c];
      const long h = o.image_size.empty() ? static_cast<long>(std::ceil(2 * cy)) : o.image_size[2 * c + 1];
      out << c + 1 << " " << model_name(p.model) << " " << w << " " << h;
      for (std::size_t a = 0; a < np; ++a) out << " " << fmt(k[a]);
      out << "\n";
    }
  }
  {
    auto out = open_out(path_in(dir, "images.txt"));
    std::size_t n_exp = 0;
    for (char e : exported) n_exp += e != 0;
    out << "# Image list with two lines of data per image:\n"
           "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
           "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        << "# Number of images: " << n_exp << "\n";
    for (std::size_t i = 0; i < n_img; ++i) {
      if (!exported[i]) continue;
      const double* q = &p.poses[7 * i];
      char name[32];
      std::snprintf(name, sizeof(name), "image_%06zu", i);
      out << i + 1;
      for (int a = 0; a < 7; ++a) out << " " << fmt(q[a]);
      out << " " << p.intrinsics_of_image(i) + 1 << " " << (o.image_names.empty() ? std::string(name) : o.image_names[i])
          << "\n";
      for (std::size_t q2 = 0; q2 < pts2d[i].size(); ++q2) {
        const auto& v = pts2d[i][q2];
        out << (q2 ? " " : "") << fmt(v[0]) << " " << fmt(v[1]) << " " << static_cast<long>(v[2]);
      }
      out << "\n";
    }
  }
  {
    auto out = open_out(path_in(dir, "points3D.txt"));
    std::vector<std::vector<std::size_t>> by_point(n_pts);
    for (std::size_t k = 0; k < n_obs; ++k)
      if (idx2d[k] >= 0) by_point[static_cast<std::size_t>(p.obs_point[k])].push_back(k);
    std::size_t n_exp = 0;
    for (const auto& b : by_point) n_exp += !b.empty();
    out << "# 3D point list with one line of data per point:\n"
           "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        << "# Number of points: " << n_exp << "\n";
    for (std::size_t j = 0; j < n_pts; ++j) {
      if (by_point[j].empty()) continue;
      double err = 0;
      for (auto k : by_point[j]) err += rep.reprojection_error[k];
      err /= static_cast<double>(by_point[j].size());
      out << j + 1 << " " << fmt(p.points[3 * j]) << " " << fmt(p.points[3 * j + 1]) << " " << fmt(p.points[3 * j + 2]);
      for (int c = 0; c < 3; ++c) out << " " << (o.point_color.empty() ? 128 : int(o.point_color[3 * j + static_cast<std::size_t>(c)]));
      out << " " << fmt(err);
      for (auto k : by_point[j]) out << " " << p.obs_image[k] + 1 << " " << idx2d[k];
      out << "\n";
    }
  }
}

void write_result(const std::string& path, const Problem& p, const Covariance* cov, const ResidualReport* rep,
                  const std::vector<std::int32_t>* obs_keypoint) {
  Problem q = p;
  std::vector<RawSection> keep;
  for (auto& s : q.extra)
    if (s.name.rfind("result.", 0) != 0) keep.push_back(std::move(s));
  q.extra = std::move(keep);
  const std::size_t n_img = p.num_images(), n_obs = p.num_observations();
  if (cov) {
    if (cov->pose.size() != 36 * n_img) throw std::runtime_error("write_result: covariance is not of this problem");
    q.extra.push_back(section("result.pose_covariance", cov->pose, {n_img, 6, 6}));
    q.extra.push_back(section("result.centre_covariance", cov->centre, {n_img, 3, 3}));
    if (!cov->points.empty()) {
      q.extra.push_back(section_i32("result.point_covariance.index", cov->points));
      q.extra.push_back(section("result.point_covariance", cov->point, {cov->points.size(), 3, 3}));
    }
  }
  if (rep) {
    if (rep->reprojection_error.size() != n_obs) throw std::runtime_error("write_result: report is not of this problem");
    const std::vector<double> costs = {rep->reprojection_cost, rep->relative_prior_cost, rep->global_prior_cost,
                                       rep->gravity_cost,      rep->line_cost,           rep->depth_cost};
    q.extra.push_back(section("result.cost_by_type", costs, {costs.size()}));
    q.extra.push_back(section("result.reprojection_error", rep->reprojection_error, {n_obs}));
    auto add = [&](const char* name, const std::vector<double>& v) {
      if (!v.empty()) q.extra.push_back(section(name, v, {v.size()}));
    };
    add("result.relative_prior_weight", rep->relative_prior_weight);
    add("result.global_prior_weight", rep->global_prior_weight);
    add("result.gravity_weight", rep->gravity_weight);
    add("result.line_angle", rep->line_angle);
    add("result.depth_refined", rep->depth_refined);
    add("result.depth_corrected", rep->depth_corrected);
  }
  if (obs_keypoint) {
    if (obs_keypoint->size() != n_obs) throw std::runtime_error("write_result: one keypoint per observation");
    q.extra.push_back(section_i32("result.obs_keypoint", *obs_keypoint));
  }
  write_problem(path, q);
}

}  // namespace jba
