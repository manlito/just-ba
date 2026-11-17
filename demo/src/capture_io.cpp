#include "capture_io.h"

#include <map>
#include <stdexcept>

#include "container.h"
#include "jba/camera_model.h"

namespace demo {
namespace {

using Sections = std::map<std::string, Section>;

[[noreturn]] void fail(const std::string& path, const std::string& what) {
  throw std::runtime_error(path + ": " + what);
}

const Section* find(const Sections& s, const std::string& name) {
  const auto it = s.find(name);
  return it == s.end() ? nullptr : &it->second;
}

// The section, checked for its data type and shape (-1: any extent).
template <class T>
std::vector<T> get(const Sections& s, const std::string& path, const std::string& name,
                   std::vector<std::int64_t> shape, bool required = true) {
  const Section* sec = find(s, name);
  if (!sec) {
    if (required) fail(path, "missing section " + name);
    return {};
  }
  bool ok = sec->shape.size() == shape.size();
  for (std::size_t d = 0; ok && d < shape.size(); ++d)
    ok = shape[d] < 0 || sec->shape[d] == static_cast<std::uint64_t>(shape[d]);
  if (!ok) fail(path, "section " + name + " has the wrong shape");
  try {
    return sec->as<T>();
  } catch (const std::exception&) {
    fail(path, "section " + name + " has the wrong data type");
  }
}

std::int64_t dim(const Sections& s, const std::string& name, std::size_t d) {
  const Section* sec = find(s, name);
  return sec && sec->shape.size() > d ? static_cast<std::int64_t>(sec->shape[d]) : -1;
}

}  // namespace

CaptureFile read_capture(const std::string& path) {
  const Sections s = read_sections(path);
  CaptureFile f;
  Capture& c = f.capture;

  const auto model = get<std::int32_t>(s, path, "camera.model", {1});
  const auto m = jba::model_from_id(model[0]);
  if (!m) fail(path, "unknown camera model " + std::to_string(model[0]));
  c.model = *m;
  const std::int64_t np = jba::num_params(c.model);
  c.intrinsics = get<double>(s, path, "intrinsics", {-1, np});
  const std::int64_t blocks = static_cast<std::int64_t>(c.intrinsics.size()) / np;
  if (blocks < 1) fail(path, "no intrinsics");
  f.image_size = get<std::int32_t>(s, path, "camera.size", {blocks, 2}, false);

  const auto kp_begin = get<std::int64_t>(s, path, "keypoint.begin", {-1});
  if (kp_begin.size() < 2 || kp_begin.front() != 0) fail(path, "keypoint.begin needs at least one image, from 0");
  const std::int64_t n_img = static_cast<std::int64_t>(kp_begin.size()) - 1;
  const auto kp_xy = get<double>(s, path, "keypoint.xy", {kp_begin.back(), 2});
  c.keypoints.resize(static_cast<std::size_t>(n_img));
  for (std::int64_t i = 0; i < n_img; ++i) {
    if (kp_begin[i + 1] < kp_begin[i]) fail(path, "keypoint.begin decreases");
    c.keypoints[i].assign(kp_xy.begin() + 2 * kp_begin[i], kp_xy.begin() + 2 * kp_begin[i + 1]);
  }
  c.image_intrinsics = get<std::int32_t>(s, path, "image.intrinsics", {n_img}, blocks > 1);

  const auto names = get<std::uint8_t>(s, path, "image.names", {-1}, false);
  if (!names.empty()) {
    std::string cur;
    for (std::uint8_t ch : names) {
      if (ch == '\n') {
        f.names.push_back(cur);
        cur.clear();
      } else {
        cur += static_cast<char>(ch);
      }
    }
    f.names.push_back(cur);
    if (static_cast<std::int64_t>(f.names.size()) != n_img) fail(path, "image.names needs one name per image");
  }

  const auto pair_images = get<std::int32_t>(s, path, "pair.images", {-1, 2});
  const std::int64_t n_pairs = static_cast<std::int64_t>(pair_images.size()) / 2;
  const auto pair_begin = get<std::int64_t>(s, path, "pair.begin", {n_pairs + 1});
  const auto match = get<std::int32_t>(s, path, "pair.match", {pair_begin.back(), 2});
  if (pair_begin.front() != 0) fail(path, "pair.begin must start at 0");
  for (std::int64_t r = 0; r < n_pairs; ++r) {
    if (pair_begin[r + 1] < pair_begin[r]) fail(path, "pair.begin decreases");
    Pair p;
    p.a = pair_images[2 * r];
    p.b = pair_images[2 * r + 1];
    for (std::int64_t k = pair_begin[r]; k < pair_begin[r + 1]; ++k) {
      p.ka.push_back(match[2 * k]);
      p.kb.push_back(match[2 * k + 1]);
    }
    c.pairs.push_back(std::move(p));
  }

  if (find(s, "reference.registered")) {
    Reference& ref = f.reference;
    ref.intrinsics = get<double>(s, path, "reference.intrinsics", {blocks, np}, false);
    const auto reg = get<std::uint8_t>(s, path, "reference.registered", {n_img});
    ref.registered.assign(reg.begin(), reg.end());
    ref.poses = get<double>(s, path, "reference.pose", {n_img, 7});
    ref.points = get<double>(s, path, "reference.point.xyz", {-1, 3});
    const std::int64_t n_obs = dim(s, "reference.obs.image", 0);
    ref.obs_image = get<std::int32_t>(s, path, "reference.obs.image", {n_obs});
    ref.obs_point = get<std::int32_t>(s, path, "reference.obs.point", {n_obs});
    ref.obs_keypoint = get<std::int32_t>(s, path, "reference.obs.keypoint", {n_obs});
    const std::int64_t n_pts = static_cast<std::int64_t>(ref.points.size()) / 3;
    for (std::int64_t k = 0; k < n_obs; ++k) {
      const std::int32_t i = ref.obs_image[k];
      if (i < 0 || i >= n_img || !ref.registered[i] || ref.obs_point[k] < 0 || ref.obs_point[k] >= n_pts ||
          ref.obs_keypoint[k] < 0 || static_cast<std::size_t>(ref.obs_keypoint[k]) >= c.keypoints[i].size() / 2)
        fail(path, "a reference observation names an image, point or keypoint that does not exist");
    }
  }
  return f;
}

void write_capture(const std::string& path, const CaptureFile& f) {
  const Capture& c = f.capture;
  const std::uint64_t np = static_cast<std::uint64_t>(jba::num_params(c.model));
  const std::uint64_t blocks = c.intrinsics.size() / np, n_img = c.num_images();
  std::vector<std::pair<std::string, Section>> s;
  s.emplace_back("camera.model", make_section(std::vector<std::int32_t>{static_cast<std::int32_t>(c.model)}, {1}));
  s.emplace_back("intrinsics", make_section(c.intrinsics, {blocks, np}));
  if (!f.image_size.empty()) s.emplace_back("camera.size", make_section(f.image_size, {blocks, 2}));
  if (!c.image_intrinsics.empty()) s.emplace_back("image.intrinsics", make_section(c.image_intrinsics, {n_img}));
  if (!f.names.empty()) {
    std::vector<std::uint8_t> bytes;
    for (std::size_t i = 0; i < f.names.size(); ++i) {
      if (i) bytes.push_back('\n');
      bytes.insert(bytes.end(), f.names[i].begin(), f.names[i].end());
    }
    s.emplace_back("image.names", make_section(bytes, {bytes.size()}));
  }
  std::vector<std::int64_t> kp_begin = {0};
  std::vector<double> kp_xy;
  for (const auto& k : c.keypoints) {
    kp_xy.insert(kp_xy.end(), k.begin(), k.end());
    kp_begin.push_back(static_cast<std::int64_t>(kp_xy.size() / 2));
  }
  s.emplace_back("keypoint.begin", make_section(kp_begin, {n_img + 1}));
  s.emplace_back("keypoint.xy", make_section(kp_xy, {kp_xy.size() / 2, 2}));
  std::vector<std::int32_t> pair_images, match;
  std::vector<std::int64_t> pair_begin = {0};
  for (const Pair& p : c.pairs) {
    pair_images.insert(pair_images.end(), {p.a, p.b});
    for (std::size_t k = 0; k < p.ka.size(); ++k) match.insert(match.end(), {p.ka[k], p.kb[k]});
    pair_begin.push_back(static_cast<std::int64_t>(match.size() / 2));
  }
  s.emplace_back("pair.images", make_section(pair_images, {c.pairs.size(), 2}));
  s.emplace_back("pair.begin", make_section(pair_begin, {c.pairs.size() + 1}));
  s.emplace_back("pair.match", make_section(match, {match.size() / 2, 2}));
  const Reference& r = f.reference;
  if (!r.empty()) {
    if (!r.intrinsics.empty()) s.emplace_back("reference.intrinsics", make_section(r.intrinsics, {blocks, np}));
    s.emplace_back("reference.registered",
                   make_section(std::vector<std::uint8_t>(r.registered.begin(), r.registered.end()), {n_img}));
    s.emplace_back("reference.pose", make_section(r.poses, {n_img, 7}));
    s.emplace_back("reference.point.xyz", make_section(r.points, {r.points.size() / 3, 3}));
    s.emplace_back("reference.obs.image", make_section(r.obs_image, {r.obs_image.size()}));
    s.emplace_back("reference.obs.point", make_section(r.obs_point, {r.obs_point.size()}));
    s.emplace_back("reference.obs.keypoint", make_section(r.obs_keypoint, {r.obs_keypoint.size()}));
  }
  write_sections(path, s);
}

}  // namespace demo
