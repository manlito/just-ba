#include "jba/problem_io.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string_view>

#include "jba/geometry.h"
#include "jba/loss.h"

static_assert(std::endian::native == std::endian::little,
              "the problem file format is little-endian; big-endian hosts are not supported");

namespace jba {
namespace {

// ---------------------------------------------------------------------------
// Container layer: magic, version, then named typed arrays (document 11).

constexpr char kMagic[8] = {'\x89', 'J', 'B', 'A', '\r', '\n', '\x1a', '\n'};
constexpr std::uint32_t kVersion = 1;

enum DType : std::uint8_t { kU8 = 1, kI32 = 2, kI64 = 3, kF32 = 4, kF64 = 5 };

std::size_t dtype_size(std::uint8_t dtype) {
  switch (dtype) {
    case kU8: return 1;
    case kI32: return 4;
    case kF32: return 4;
    case kI64: return 8;
    case kF64: return 8;
    default: return 0;
  }
}

template <typename T> constexpr std::uint8_t dtype_of();
template <> constexpr std::uint8_t dtype_of<std::uint8_t>() { return kU8; }
template <> constexpr std::uint8_t dtype_of<std::int32_t>() { return kI32; }
template <> constexpr std::uint8_t dtype_of<double>() { return kF64; }

[[noreturn]] void fail(const std::string& path, const std::string& what) {
  throw std::runtime_error(path + ": " + what);
}

class Writer {
 public:
  explicit Writer(std::vector<std::uint8_t>& out) : out_(out) {}

  void raw(const void* data, std::size_t n) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    out_.insert(out_.end(), p, p + n);
  }
  template <typename T> void value(T v) { raw(&v, sizeof(T)); }
  void pad() {
    while (out_.size() % 8 != 0) out_.push_back(0);
  }

  void section(const std::string& name, std::uint8_t dtype, const std::vector<std::uint64_t>& shape,
               const void* data, std::size_t bytes) {
    value<std::uint32_t>(static_cast<std::uint32_t>(name.size()));
    raw(name.data(), name.size());
    value<std::uint8_t>(dtype);
    value<std::uint8_t>(static_cast<std::uint8_t>(shape.size()));
    value<std::uint16_t>(0);
    for (auto d : shape) value<std::uint64_t>(d);
    pad();
    raw(data, bytes);
    pad();
    ++count_;
  }

  template <typename T>
  void array(const std::string& name, const std::vector<T>& v, std::uint64_t cols) {
    std::vector<std::uint64_t> shape;
    if (cols == 1) {
      shape = {v.size()};
    } else {
      shape = {v.size() / cols, cols};
    }
    section(name, dtype_of<T>(), shape, v.data(), v.size() * sizeof(T));
  }

  template <typename T>
  void array_shape(const std::string& name, const std::vector<T>& v, const std::vector<std::uint64_t>& tail) {
    std::uint64_t per = 1;
    for (auto d : tail) per *= d;
    std::vector<std::uint64_t> shape = {v.size() / per};
    shape.insert(shape.end(), tail.begin(), tail.end());
    section(name, dtype_of<T>(), shape, v.data(), v.size() * sizeof(T));
  }

  std::uint32_t count() const { return count_; }

 private:
  std::vector<std::uint8_t>& out_;
  std::uint32_t count_ = 0;
};

class Reader {
 public:
  Reader(const std::string& path, const std::vector<std::uint8_t>& data) : path_(path), data_(data) {}

  bool done() const { return pos_ >= data_.size(); }

  void raw(void* dst, std::size_t n) {
    if (n > data_.size() - pos_) fail(path_, "file is truncated");
    std::memcpy(dst, data_.data() + pos_, n);
    pos_ += n;
  }
  template <typename T> T value() {
    T v;
    raw(&v, sizeof(T));
    return v;
  }
  void align() {
    const std::size_t aligned = (pos_ + 7) / 8 * 8;
    if (aligned > data_.size()) fail(path_, "file is truncated");
    pos_ = aligned;
  }

  RawSection section() {
    RawSection s;
    const auto name_len = value<std::uint32_t>();
    if (name_len > 256) fail(path_, "section name too long; file is corrupt");
    s.name.resize(name_len);
    raw(s.name.data(), name_len);
    s.dtype = value<std::uint8_t>();
    const auto ndim = value<std::uint8_t>();
    value<std::uint16_t>();
    const std::size_t elem = dtype_size(s.dtype);
    if (elem == 0) fail(path_, "section '" + s.name + "' has unknown dtype " + std::to_string(s.dtype));
    std::uint64_t count = 1;
    for (int i = 0; i < ndim; ++i) {
      const auto d = value<std::uint64_t>();
      s.shape.push_back(d);
      if (d != 0 && count > (std::uint64_t(1) << 40) / d) fail(path_, "section '" + s.name + "' is too large");
      count *= d;
    }
    align();
    const std::uint64_t bytes = count * elem;
    if (bytes > data_.size() - pos_) fail(path_, "section '" + s.name + "' is truncated");
    s.bytes.assign(data_.begin() + static_cast<std::ptrdiff_t>(pos_),
                   data_.begin() + static_cast<std::ptrdiff_t>(pos_ + bytes));
    pos_ += bytes;
    align();
    return s;
  }

 private:
  const std::string& path_;
  const std::vector<std::uint8_t>& data_;
  std::size_t pos_ = 0;
};

// Reads a section of shape [n] (tail empty) or [n, tail...].
template <typename T>
void take_shape(const std::string& path, const RawSection& s, const std::vector<std::uint64_t>& tail,
                std::vector<T>& out) {
  if (s.dtype != dtype_of<T>()) fail(path, "section '" + s.name + "' has the wrong dtype");
  bool ok = s.shape.size() == 1 + tail.size();
  for (std::size_t d = 0; ok && d < tail.size(); ++d) ok = s.shape[1 + d] == tail[d];
  if (!ok) fail(path, "section '" + s.name + "' has the wrong shape");
  out.resize(s.bytes.size() / sizeof(T));
  if (!out.empty()) std::memcpy(out.data(), s.bytes.data(), s.bytes.size());
}

template <typename T>
void take(const std::string& path, const RawSection& s, std::uint64_t cols, std::vector<T>& out) {
  take_shape(path, s, cols == 1 ? std::vector<std::uint64_t>{} : std::vector<std::uint64_t>{cols}, out);
}

std::vector<std::uint8_t> read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error(path + ": cannot open");
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in), {});
}

}  // namespace

// ---------------------------------------------------------------------------

namespace {

bool unit_quaternion(const double* q) {
  const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  return std::abs(n - 1.0) < 1e-9;
}

// Unit within the quaternion check's tolerance (NaN fails).
bool unit_vector(const double* v) {
  const double n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  return std::abs(n - 1.0) < 1e-9;
}

// Weights of one kind of prior: at most one of sigma and sqrt_info; sigmas positive
// (+inf allowed: that part is left out); whitening finite.
std::string validate_weights(const char* kind, std::size_t n, const std::vector<double>& sigma,
                             const std::vector<double>& sqrt_info) {
  const std::string k(kind);
  if (!sigma.empty() && !sqrt_info.empty()) return k + " priors may have sigma or sqrt_info, not both";
  if (!sigma.empty() && sigma.size() != 2 * n) return k + " prior sigma must have 2 values per prior";
  if (!sqrt_info.empty() && sqrt_info.size() != 36 * n) return k + " prior sqrt_info must have 36 values per prior";
  for (std::size_t a = 0; a < sigma.size(); ++a) {
    if (!(sigma[a] > 0)) return k + " prior " + std::to_string(a / 2) + " has a non-positive sigma";
  }
  for (std::size_t a = 0; a < sqrt_info.size(); ++a) {
    if (!std::isfinite(sqrt_info[a])) return k + " prior " + std::to_string(a / 36) + " has a non-finite sqrt_info";
  }
  return {};
}

std::string validate_priors(const Problem& p) {
  const std::size_t n_img = p.num_images();
  const std::size_t n_rel = p.num_relative_priors(), n_glob = p.num_global_priors();
  if (p.rel_images.size() % 2 != 0) return "rel_images must hold 2 values per relative prior";
  if (p.rel_pose.size() != 7 * n_rel) return "rel_pose must hold 7 values per relative prior";
  for (std::size_t k = 0; k < n_rel; ++k) {
    const auto i = p.rel_images[2 * k], j = p.rel_images[2 * k + 1];
    if (i < 0 || j < 0 || static_cast<std::size_t>(i) >= n_img || static_cast<std::size_t>(j) >= n_img)
      return "relative prior " + std::to_string(k) + " refers to a missing image";
    if (i == j) return "relative prior " + std::to_string(k) + " ties an image to itself";
    if (!unit_quaternion(&p.rel_pose[7 * k])) return "relative prior " + std::to_string(k) + " has a non-unit quaternion";
  }
  if (auto err = validate_weights("relative", n_rel, p.rel_sigma, p.rel_sqrt_info); !err.empty()) return err;

  if (p.glob_pose.size() != 7 * n_glob) return "glob_pose must hold 7 values per global prior";
  for (std::size_t a = 0; a < p.rel_pose.size(); ++a)
    if (!std::isfinite(p.rel_pose[a])) return "relative prior " + std::to_string(a / 7) + " has a value that is not finite";
  for (std::size_t a = 0; a < p.glob_pose.size(); ++a)
    if (!std::isfinite(p.glob_pose[a])) return "global prior " + std::to_string(a / 7) + " has a value that is not finite";
  for (std::size_t a = 0; a < p.set_transform.size(); ++a)
    if (!std::isfinite(p.set_transform[a])) return "prior set " + std::to_string(a / 8) + " has a value that is not finite";
  if (!p.glob_set.empty() && p.glob_set.size() != n_glob) return "glob_set must have one entry per global prior";
  const std::size_t n_sets = p.num_prior_sets();
  for (std::size_t k = 0; k < n_glob; ++k) {
    if (p.glob_image[k] < 0 || static_cast<std::size_t>(p.glob_image[k]) >= n_img)
      return "global prior " + std::to_string(k) + " refers to a missing image";
    const auto s = p.set_of_global_prior(k);
    if (s < 0 || static_cast<std::size_t>(s) >= n_sets)
      return "global prior " + std::to_string(k) + " refers to a missing prior set";
    if (!unit_quaternion(&p.glob_pose[7 * k])) return "global prior " + std::to_string(k) + " has a non-unit quaternion";
  }
  if (auto err = validate_weights("global", n_glob, p.glob_sigma, p.glob_sqrt_info); !err.empty()) return err;

  if (!p.gravity_world.empty()) {
    if (p.gravity_world.size() != 3) return "gravity_world must hold 3 values";
    if (!unit_vector(p.gravity_world.data())) return "gravity_world is not a unit vector";
  }
  const std::size_t n_grav = p.num_gravity_priors();
  if (p.grav_dir.size() != 3 * n_grav) return "grav_dir must hold 3 values per gravity prior";
  if (!p.grav_sigma.empty() && p.grav_sigma.size() != n_grav) return "grav_sigma must have one entry per gravity prior";
  for (std::size_t k = 0; k < n_grav; ++k) {
    if (p.grav_image[k] < 0 || static_cast<std::size_t>(p.grav_image[k]) >= n_img)
      return "gravity prior " + std::to_string(k) + " refers to a missing image";
    if (!unit_vector(&p.grav_dir[3 * k])) return "gravity prior " + std::to_string(k) + " is not a unit vector";
    if (!p.grav_sigma.empty() && !(p.grav_sigma[k] > 0 && std::isfinite(p.grav_sigma[k])))
      return "gravity prior " + std::to_string(k) + " has a sigma that is not positive and finite";
  }
  for (auto r : p.set_relation) {
    if (r != FrameRelation::Identity && r != FrameRelation::Rigid && r != FrameRelation::Similarity)
      return "unknown prior set relation " + std::to_string(static_cast<int>(r));
  }
  if (!p.set_transform.empty()) {
    if (p.set_transform.size() != 8 * n_sets) return "set_transform must hold 8 values per prior set";
    for (std::size_t s = 0; s < n_sets; ++s) {
      if (!unit_quaternion(&p.set_transform[8 * s])) return "prior set " + std::to_string(s) + " has a non-unit quaternion";
      if (!(p.set_transform[8 * s + 7] > 0)) return "prior set " + std::to_string(s) + " has a non-positive scale";
      if (p.relation_of_set(s) == FrameRelation::Identity) {
        static constexpr double kIdentity[8] = {1, 0, 0, 0, 0, 0, 0, 1};
        for (int a = 0; a < 8; ++a)
          if (p.set_transform[8 * s + static_cast<std::size_t>(a)] != kIdentity[a])
            return "prior set " + std::to_string(s) + " is the working frame but its transform is not the identity";
      }
      if (p.relation_of_set(s) == FrameRelation::Rigid && p.set_transform[8 * s + 7] != 1)
        return "prior set " + std::to_string(s) + " is rigid but its scale is not 1";
    }
  }
  return {};
}

}  // namespace

void gravity_direction(const Problem& p, double* g) {
  if (p.gravity_world.size() == 3) {
    std::copy_n(p.gravity_world.data(), 3, g);
  } else {
    g[0] = 0;
    g[1] = -1;
    g[2] = 0;
  }
}

std::string validate(const Problem& p) {
  const int np = num_params(p.model);
  if (p.intrinsics.empty() || p.intrinsics.size() % static_cast<std::size_t>(np) != 0)
    return "intrinsics must hold a positive multiple of " + std::to_string(np) + " values";
  const std::size_t blocks = p.num_intrinsics_blocks();
  if (p.poses.size() % 7 != 0) return "poses must hold 7 values per image";
  if (p.points.size() % 3 != 0) return "points must hold 3 values per point";
  const std::size_t n_img = p.num_images(), n_pts = p.num_points(), n_obs = p.num_observations();
  if (!p.intrinsics_free.empty() && p.intrinsics_free.size() != blocks)
    return "intrinsics_free must have one entry per intrinsics block";
  if (!p.image_intrinsics.empty() && p.image_intrinsics.size() != n_img)
    return "image_intrinsics must have one entry per image";
  if (p.image_intrinsics.empty() && blocks != 1)
    return "image_intrinsics is required when there is more than one intrinsics block";
  if (!p.image_fixed.empty() && p.image_fixed.size() != n_img) return "image_fixed must have one entry per image";
  if (!p.point_fixed.empty() && p.point_fixed.size() != n_pts) return "point_fixed must have one entry per point";
  if (p.obs_point.size() != n_obs || p.obs_xy.size() != 2 * n_obs)
    return "observation arrays have different lengths";
  if (!p.obs_sigma.empty() && p.obs_sigma.size() != n_obs) return "obs_sigma must have one entry per observation";
  auto first_not_finite = [](const std::vector<double>& v) {
    std::size_t i = 0;
    while (i < v.size() && std::isfinite(v[i])) ++i;
    return i;
  };
  if (const auto i = first_not_finite(p.intrinsics); i < p.intrinsics.size())
    return "intrinsics block " + std::to_string(i / static_cast<std::size_t>(np)) + " has a value that is not finite";
  if (const auto i = first_not_finite(p.poses); i < p.poses.size())
    return "image " + std::to_string(i / 7) + " has a pose value that is not finite";
  if (const auto i = first_not_finite(p.points); i < p.points.size())
    return "point " + std::to_string(i / 3) + " has a coordinate that is not finite";
  if (const auto i = first_not_finite(p.obs_xy); i < p.obs_xy.size())
    return "observation " + std::to_string(i / 2) + " has a coordinate that is not finite";
  for (std::size_t i = 0; i < p.image_intrinsics.size(); ++i) {
    if (p.image_intrinsics[i] < 0 || static_cast<std::size_t>(p.image_intrinsics[i]) >= blocks)
      return "image " + std::to_string(i) + " refers to a missing intrinsics block";
  }
  for (std::size_t i = 0; i < n_img; ++i) {
    if (!unit_quaternion(&p.poses[7 * i])) return "image " + std::to_string(i) + " has a non-unit quaternion";
  }
  if (auto err = validate_priors(p); !err.empty()) return err;
  if (p.loss_type.size() != p.loss_scale.size()) return "loss_type and loss_scale must have the same length";
  if (p.loss_type.size() > kNumLossTypes) return "more losses than residual types";
  for (std::size_t t = 0; t < p.loss_type.size(); ++t) {
    if (p.loss_type[t] < 0 || p.loss_type[t] > static_cast<std::int32_t>(LossType::GemanMcClure))
      return "loss " + std::to_string(t) + " has an unknown type";
    if (!(p.loss_scale[t] > 0 && std::isfinite(p.loss_scale[t])))
      return "loss " + std::to_string(t) + " has a scale that is not positive and finite";
  }
  for (std::size_t k = 0; k < n_obs; ++k) {
    if (p.obs_image[k] < 0 || static_cast<std::size_t>(p.obs_image[k]) >= n_img)
      return "observation " + std::to_string(k) + " refers to a missing image";
    if (p.obs_point[k] < 0 || static_cast<std::size_t>(p.obs_point[k]) >= n_pts)
      return "observation " + std::to_string(k) + " refers to a missing point";
    if (!p.obs_sigma.empty() && !(p.obs_sigma[k] > 0 && std::isfinite(p.obs_sigma[k])))
      return "observation " + std::to_string(k) + " has a sigma that is not positive and finite";
  }
  return {};
}

void write_problem(const std::string& path, const Problem& p) {
  if (auto err = validate(p); !err.empty()) throw std::runtime_error(path + ": invalid problem: " + err);
  const auto np = static_cast<std::uint64_t>(num_params(p.model));

  std::vector<std::uint8_t> body;
  Writer w(body);
  const std::vector<std::int32_t> model = {static_cast<std::int32_t>(p.model)};
  w.array("camera.model", model, 1);
  w.array("intrinsics", p.intrinsics, np);
  if (!p.intrinsics_free.empty()) w.array("intrinsics.free", p.intrinsics_free, 1);
  w.array("image.pose", p.poses, 7);
  if (!p.image_intrinsics.empty()) w.array("image.intrinsics", p.image_intrinsics, 1);
  if (!p.image_fixed.empty()) w.array("image.fixed", p.image_fixed, 1);
  w.array("point.xyz", p.points, 3);
  if (!p.point_fixed.empty()) w.array("point.fixed", p.point_fixed, 1);
  w.array("obs.image", p.obs_image, 1);
  w.array("obs.point", p.obs_point, 1);
  w.array("obs.xy", p.obs_xy, 2);
  if (!p.obs_sigma.empty()) w.array("obs.sigma", p.obs_sigma, 1);
  if (p.num_relative_priors() > 0) {
    w.array("prior.relative.images", p.rel_images, 2);
    w.array("prior.relative.pose", p.rel_pose, 7);
    if (!p.rel_sigma.empty()) w.array("prior.relative.sigma", p.rel_sigma, 2);
    if (!p.rel_sqrt_info.empty()) w.array_shape("prior.relative.sqrt_info", p.rel_sqrt_info, {6, 6});
  }
  if (p.num_global_priors() > 0) {
    w.array("prior.global.image", p.glob_image, 1);
    if (!p.glob_set.empty()) w.array("prior.global.set", p.glob_set, 1);
    w.array("prior.global.pose", p.glob_pose, 7);
    if (!p.glob_sigma.empty()) w.array("prior.global.sigma", p.glob_sigma, 2);
    if (!p.glob_sqrt_info.empty()) w.array_shape("prior.global.sqrt_info", p.glob_sqrt_info, {6, 6});
  }
  if (!p.set_relation.empty()) {
    std::vector<std::int32_t> rel(p.set_relation.size());
    for (std::size_t a = 0; a < rel.size(); ++a) rel[a] = static_cast<std::int32_t>(p.set_relation[a]);
    w.array("prior.set.relation", rel, 1);
  }
  if (!p.set_transform.empty()) w.array("prior.set.transform", p.set_transform, 8);
  if (!p.gravity_world.empty()) w.array("gravity.world", p.gravity_world, 1);
  if (p.num_gravity_priors() > 0) {
    w.array("prior.gravity.image", p.grav_image, 1);
    w.array("prior.gravity.direction", p.grav_dir, 3);
    if (!p.grav_sigma.empty()) w.array("prior.gravity.sigma", p.grav_sigma, 1);
  }
  if (!p.loss_type.empty()) {
    w.array("loss.type", p.loss_type, 1);
    w.array("loss.scale", p.loss_scale, 1);
  }
  // A name may appear once (document 11); the reader refuses anything else.
  static constexpr std::string_view kKnown[] = {
      "camera.model",          "intrinsics",          "intrinsics.free",       "image.pose",
      "image.intrinsics",      "image.fixed",         "point.xyz",             "point.fixed",
      "obs.image",             "obs.point",           "obs.xy",                "obs.sigma",
      "prior.relative.images", "prior.relative.pose", "prior.relative.sigma",  "prior.relative.sqrt_info",
      "prior.global.image",    "prior.global.set",    "prior.global.pose",     "prior.global.sigma",
      "prior.global.sqrt_info", "prior.set.relation", "prior.set.transform", "loss.type",
      "loss.scale",            "gravity.world",       "prior.gravity.image",   "prior.gravity.direction",
      "prior.gravity.sigma"};
  for (std::size_t i = 0; i < p.extra.size(); ++i) {
    const auto& s = p.extra[i];
    bool repeated = std::find(std::begin(kKnown), std::end(kKnown), s.name) != std::end(kKnown);
    for (std::size_t j = 0; j < i && !repeated; ++j) repeated = p.extra[j].name == s.name;
    if (repeated) throw std::runtime_error(path + ": section '" + s.name + "' appears more than once");
    w.section(s.name, s.dtype, s.shape, s.bytes.data(), s.bytes.size());
  }

  std::vector<std::uint8_t> file;
  Writer h(file);
  h.raw(kMagic, sizeof(kMagic));
  h.value<std::uint32_t>(kVersion);
  h.value<std::uint32_t>(w.count());
  file.insert(file.end(), body.begin(), body.end());

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error(path + ": cannot open for writing");
  out.write(reinterpret_cast<const char*>(file.data()), static_cast<std::streamsize>(file.size()));
  if (!out) throw std::runtime_error(path + ": write failed");
}

Problem read_problem(const std::string& path) {
  const auto data = read_file(path);
  Reader r(path, data);
  char magic[8];
  r.raw(magic, sizeof(magic));
  if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) fail(path, "not a just-ba problem file");
  const auto version = r.value<std::uint32_t>();
  if (version != kVersion)
    fail(path, "unsupported format version " + std::to_string(version) + " (this build reads 1)");
  const auto count = r.value<std::uint32_t>();

  Problem p;
  bool have_model = false, have_intr = false, have_pose = false, have_pts = false;
  bool have_oi = false, have_op = false, have_oxy = false;
  std::vector<RawSection> sections;
  for (std::uint32_t i = 0; i < count; ++i) sections.push_back(r.section());
  if (!r.done()) fail(path, "trailing data after the last section");
  {
    std::vector<std::string_view> names;
    for (const auto& s : sections) names.push_back(s.name);
    std::sort(names.begin(), names.end());
    const auto dup = std::adjacent_find(names.begin(), names.end());
    if (dup != names.end()) fail(path, "section '" + std::string(*dup) + "' appears more than once");
  }

  // The model decides the intrinsics width, so read it first.
  for (const auto& s : sections) {
    if (s.name != "camera.model") continue;
    std::vector<std::int32_t> m;
    take(path, s, 1, m);
    if (m.size() != 1) fail(path, "camera.model must hold one value");
    auto model = model_from_id(m[0]);
    if (!model) fail(path, "unknown camera model id " + std::to_string(m[0]));
    p.model = *model;
    have_model = true;
  }
  if (!have_model) fail(path, "missing section 'camera.model'");
  const auto np = static_cast<std::uint64_t>(num_params(p.model));

  for (auto& s : sections) {
    const std::string& n = s.name;
    if (n == "camera.model") continue;
    if (n == "intrinsics") { take(path, s, np, p.intrinsics); have_intr = true; }
    else if (n == "intrinsics.free") take(path, s, 1, p.intrinsics_free);
    else if (n == "image.pose") { take(path, s, 7, p.poses); have_pose = true; }
    else if (n == "image.intrinsics") take(path, s, 1, p.image_intrinsics);
    else if (n == "image.fixed") take(path, s, 1, p.image_fixed);
    else if (n == "point.xyz") { take(path, s, 3, p.points); have_pts = true; }
    else if (n == "point.fixed") take(path, s, 1, p.point_fixed);
    else if (n == "obs.image") { take(path, s, 1, p.obs_image); have_oi = true; }
    else if (n == "obs.point") { take(path, s, 1, p.obs_point); have_op = true; }
    else if (n == "obs.xy") { take(path, s, 2, p.obs_xy); have_oxy = true; }
    else if (n == "obs.sigma") take(path, s, 1, p.obs_sigma);
    else if (n == "prior.relative.images") take(path, s, 2, p.rel_images);
    else if (n == "prior.relative.pose") take(path, s, 7, p.rel_pose);
    else if (n == "prior.relative.sigma") take(path, s, 2, p.rel_sigma);
    else if (n == "prior.relative.sqrt_info") take_shape(path, s, {6, 6}, p.rel_sqrt_info);
    else if (n == "prior.global.image") take(path, s, 1, p.glob_image);
    else if (n == "prior.global.set") take(path, s, 1, p.glob_set);
    else if (n == "prior.global.pose") take(path, s, 7, p.glob_pose);
    else if (n == "prior.global.sigma") take(path, s, 2, p.glob_sigma);
    else if (n == "prior.global.sqrt_info") take_shape(path, s, {6, 6}, p.glob_sqrt_info);
    else if (n == "prior.set.relation") {
      std::vector<std::int32_t> rel;
      take(path, s, 1, rel);
      p.set_relation.resize(rel.size());
      for (std::size_t a = 0; a < rel.size(); ++a) p.set_relation[a] = static_cast<FrameRelation>(rel[a]);
    }
    else if (n == "prior.set.transform") take(path, s, 8, p.set_transform);
    else if (n == "gravity.world") take(path, s, 1, p.gravity_world);
    else if (n == "prior.gravity.image") take(path, s, 1, p.grav_image);
    else if (n == "prior.gravity.direction") take(path, s, 3, p.grav_dir);
    else if (n == "prior.gravity.sigma") take(path, s, 1, p.grav_sigma);
    else if (n == "loss.type") take(path, s, 1, p.loss_type);
    else if (n == "loss.scale") take(path, s, 1, p.loss_scale);
    else p.extra.push_back(std::move(s));
  }
  if (!have_intr) fail(path, "missing section 'intrinsics'");
  if (!have_pose) fail(path, "missing section 'image.pose'");
  if (!have_pts) fail(path, "missing section 'point.xyz'");
  if (!have_oi || !have_op || !have_oxy) fail(path, "missing observation sections");
  if (auto err = validate(p); !err.empty()) fail(path, "invalid problem: " + err);
  return p;
}

// ---------------------------------------------------------------------------
// BAL reader.

namespace {

class Tokens {
 public:
  Tokens(const std::string& path, std::string text) : path_(path), text_(std::move(text)) {
    cur_ = text_.c_str();
  }
  double real() {
    char* end = nullptr;
    const double v = std::strtod(cur_, &end);
    if (end == cur_) fail(path_, "malformed BAL file");
    cur_ = end;
    return v;
  }
  long integer() {
    char* end = nullptr;
    const long v = std::strtol(cur_, &end, 10);
    if (end == cur_) fail(path_, "malformed BAL file");
    cur_ = end;
    return v;
  }

 private:
  const std::string& path_;
  std::string text_;
  const char* cur_;
};

}  // namespace

Problem read_bal(const std::string& path) {
  const auto bytes = read_file(path);
  Tokens tok(path, std::string(bytes.begin(), bytes.end()));
  const long n_cam = tok.integer(), n_pts = tok.integer(), n_obs = tok.integer();
  if (n_cam <= 0 || n_pts < 0 || n_obs < 0) fail(path, "bad BAL header");

  Problem p;
  p.model = CameraModel::Radial;
  p.obs_image.resize(static_cast<std::size_t>(n_obs));
  p.obs_point.resize(static_cast<std::size_t>(n_obs));
  p.obs_xy.resize(2 * static_cast<std::size_t>(n_obs));
  // BAL image coordinates have y up; ours have y down.
  for (long k = 0; k < n_obs; ++k) {
    p.obs_image[k] = static_cast<std::int32_t>(tok.integer());
    p.obs_point[k] = static_cast<std::int32_t>(tok.integer());
    p.obs_xy[2 * k] = tok.real();
    p.obs_xy[2 * k + 1] = -tok.real();
  }

  // BAL camera: angle-axis R, t, f, k1, k2, with P = R X + t and the camera looking
  // down -z. Rotating the camera frame by pi about x gives our convention exactly:
  // R' = F R, t' = F t, F = diag(1, -1, -1), whose quaternion is (0, 1, 0, 0).
  const double q_flip[4] = {0, 1, 0, 0};
  p.poses.resize(7 * static_cast<std::size_t>(n_cam));
  p.intrinsics.resize(5 * static_cast<std::size_t>(n_cam));
  p.intrinsics_free.assign(static_cast<std::size_t>(n_cam), kFocal | kDistortion);
  p.image_intrinsics.resize(static_cast<std::size_t>(n_cam));
  for (long i = 0; i < n_cam; ++i) {
    double c[9];
    for (double& v : c) v = tok.real();
    double q_bal[4];
    quat_exp(c, q_bal);
    double* pose = &p.poses[7 * i];
    quat_multiply(q_flip, q_bal, pose);
    quat_normalize(pose);
    pose[4] = c[3];
    pose[5] = -c[4];
    pose[6] = -c[5];
    double* k = &p.intrinsics[5 * i];
    k[0] = c[6];
    k[1] = 0;
    k[2] = 0;
    k[3] = c[7];
    k[4] = c[8];
    p.image_intrinsics[i] = static_cast<std::int32_t>(i);
  }
  p.points.resize(3 * static_cast<std::size_t>(n_pts));
  for (auto& v : p.points) v = tok.real();

  if (auto err = validate(p); !err.empty()) fail(path, "invalid BAL problem: " + err);
  return p;
}

}  // namespace jba
