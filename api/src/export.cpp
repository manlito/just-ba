#include "jba/export.h"

#include <array>
#include <bit>
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

// Little-endian binary output, COLMAP's byte order whatever the host's.
class BinaryOut {
 public:
  explicit BinaryOut(const std::string& path) : path_(path), out_(path, std::ios::binary) {
    if (!out_) throw std::runtime_error(path + ": cannot open for writing");
  }
  void u8(std::uint8_t v) { put(v); }
  void i32(std::int32_t v) { put(v); }
  void u32(std::uint32_t v) { put(v); }
  void u64(std::uint64_t v) { put(v); }
  void f64(double v) { put(std::bit_cast<std::uint64_t>(v)); }
  void str(const std::string& s) { out_.write(s.c_str(), static_cast<std::streamsize>(s.size() + 1)); }
  void close() {
    out_.close();
    if (!out_) throw std::runtime_error(path_ + ": write failed");
  }

 private:
  template <typename T>
  void put(T v) {
    if constexpr (std::endian::native == std::endian::big && sizeof(T) > 1) v = std::byteswap(v);
    out_.write(reinterpret_cast<const char*>(&v), sizeof(T));
  }
  std::string path_;
  std::ofstream out_;
};

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

  // Per point, its exported observations; per exported image and block, the size.
  std::vector<std::vector<std::size_t>> by_point(n_pts);
  for (std::size_t k = 0; k < n_obs; ++k)
    if (idx2d[k] >= 0) by_point[static_cast<std::size_t>(p.obs_point[k])].push_back(k);
  std::size_t n_img_out = 0, n_pts_out = 0;
  for (char e : exported) n_img_out += e != 0;
  for (const auto& b : by_point) n_pts_out += !b.empty();
  // Principal point: index 1, 2 for the single-focal models, 2, 3 otherwise.
  const bool single = p.model == CameraModel::SimplePinhole || p.model == CameraModel::SimpleRadial ||
                      p.model == CameraModel::Radial;
  auto size_of = [&](std::size_t c) {
    const double* k = &p.intrinsics[np * c];
    const double cx = single ? k[1] : k[2], cy = single ? k[2] : k[3];
    const long w = o.image_size.empty() ? static_cast<long>(std::ceil(2 * cx)) : o.image_size[2 * c];
    const long h = o.image_size.empty() ? static_cast<long>(std::ceil(2 * cy)) : o.image_size[2 * c + 1];
    return std::array<long, 2>{w, h};
  };
  auto name_of = [&](std::size_t i) {
    if (!o.image_names.empty()) return o.image_names[i];
    char name[32];
    std::snprintf(name, sizeof(name), "image_%06zu", i);
    return std::string(name);
  };
  auto error_of = [&](std::size_t j) {
    double err = 0;
    for (auto k : by_point[j]) err += rep.reprojection_error[k];
    return err / static_cast<double>(by_point[j].size());
  };
  auto color_of = [&](std::size_t j, int c) -> std::uint8_t {
    return o.point_color.empty() ? 128 : o.point_color[3 * j + static_cast<std::size_t>(c)];
  };

  if (o.binary) {
    // COLMAP's binary model (src/colmap/scene/reconstruction_io.cc there): little endian,
    // counts u64, ids u32 (point ids u64), names NUL-terminated, a 2-D point without a
    // 3-D point has id 2^64 - 1.
    for (const auto& n : (o.image_names.empty() ? std::vector<std::string>{} : o.image_names))
      if (n.find('\0') != std::string::npos) throw std::runtime_error("export_colmap: an image name holds a NUL");
    {
      BinaryOut out(path_in(dir, "cameras.bin"));
      out.u64(n_blk);
      for (std::size_t c = 0; c < n_blk; ++c) {
        const auto wh = size_of(c);
        out.u32(static_cast<std::uint32_t>(c + 1));
        out.i32(static_cast<std::int32_t>(p.model));
        out.u64(static_cast<std::uint64_t>(wh[0]));
        out.u64(static_cast<std::uint64_t>(wh[1]));
        for (std::size_t a = 0; a < np; ++a) out.f64(p.intrinsics[np * c + a]);
      }
      out.close();
    }
    {
      BinaryOut out(path_in(dir, "images.bin"));
      out.u64(n_img_out);
      for (std::size_t i = 0; i < n_img; ++i) {
        if (!exported[i]) continue;
        out.u32(static_cast<std::uint32_t>(i + 1));
        for (int a = 0; a < 7; ++a) out.f64(p.poses[7 * i + static_cast<std::size_t>(a)]);
        out.u32(static_cast<std::uint32_t>(p.intrinsics_of_image(i) + 1));
        out.str(name_of(i));
        out.u64(pts2d[i].size());
        for (const auto& v : pts2d[i]) {
          out.f64(v[0]);
          out.f64(v[1]);
          out.u64(v[2] < 0 ? ~std::uint64_t{0} : static_cast<std::uint64_t>(v[2]));
        }
      }
      out.close();
    }
    {
      BinaryOut out(path_in(dir, "points3D.bin"));
      out.u64(n_pts_out);
      for (std::size_t j = 0; j < n_pts; ++j) {
        if (by_point[j].empty()) continue;
        out.u64(j + 1);
        for (int a = 0; a < 3; ++a) out.f64(p.points[3 * j + static_cast<std::size_t>(a)]);
        for (int c = 0; c < 3; ++c) out.u8(color_of(j, c));
        out.f64(error_of(j));
        out.u64(by_point[j].size());
        for (auto k : by_point[j]) {
          out.u32(static_cast<std::uint32_t>(p.obs_image[k] + 1));
          out.u32(static_cast<std::uint32_t>(idx2d[k]));
        }
      }
      out.close();
    }
    return;
  }

  {
    auto out = open_out(path_in(dir, "cameras.txt"));
    out << "# Camera list with one line of data per camera:\n"
           "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        << "# Number of cameras: " << n_blk << "\n";
    for (std::size_t c = 0; c < n_blk; ++c) {
      const auto wh = size_of(c);
      out << c + 1 << " " << model_name(p.model) << " " << wh[0] << " " << wh[1];
      for (std::size_t a = 0; a < np; ++a) out << " " << fmt(p.intrinsics[np * c + a]);
      out << "\n";
    }
  }
  {
    auto out = open_out(path_in(dir, "images.txt"));
    out << "# Image list with two lines of data per image:\n"
           "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
           "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        << "# Number of images: " << n_img_out << "\n";
    for (std::size_t i = 0; i < n_img; ++i) {
      if (!exported[i]) continue;
      const double* q = &p.poses[7 * i];
      out << i + 1;
      for (int a = 0; a < 7; ++a) out << " " << fmt(q[a]);
      out << " " << p.intrinsics_of_image(i) + 1 << " " << name_of(i) << "\n";
      for (std::size_t q2 = 0; q2 < pts2d[i].size(); ++q2) {
        const auto& v = pts2d[i][q2];
        out << (q2 ? " " : "") << fmt(v[0]) << " " << fmt(v[1]) << " " << static_cast<long>(v[2]);
      }
      out << "\n";
    }
  }
  {
    auto out = open_out(path_in(dir, "points3D.txt"));
    out << "# 3D point list with one line of data per point:\n"
           "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        << "# Number of points: " << n_pts_out << "\n";
    for (std::size_t j = 0; j < n_pts; ++j) {
      if (by_point[j].empty()) continue;
      out << j + 1 << " " << fmt(p.points[3 * j]) << " " << fmt(p.points[3 * j + 1]) << " " << fmt(p.points[3 * j + 2]);
      for (int c = 0; c < 3; ++c) out << " " << int(color_of(j, c));
      out << " " << fmt(error_of(j));
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
    // The solver's types: the reprojection entry includes the intrinsics priors (D161,
    // D165).
    const std::vector<double> costs = {rep->reprojection_cost + rep->intrinsics_prior_cost + rep->intrinsics_group_cost,
                                       rep->relative_prior_cost, rep->global_prior_cost,
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
