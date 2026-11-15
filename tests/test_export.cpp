// Export (D89): COLMAP's text model read back and checked against the problem, and
// the result sections of the library's own format.

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "jba/covariance.h"
#include "jba/export.h"
#include "jba/problem_io.h"
#include "jba/report.h"
#include "synthetic.h"

namespace fs = std::filesystem;

namespace {

std::vector<std::string> data_lines(const fs::path& path) {
  std::ifstream in(path);
  std::vector<std::string> out;
  std::string line;
  while (std::getline(in, line))
    if (!line.empty() && line[0] != '#') out.push_back(line);
  return out;
}

struct Image2D {
  std::vector<double> pose;
  int camera = 0;
  std::string name;
  std::vector<std::array<double, 3>> pts;
};

void check_model(const jba::Problem& p, const fs::path& dir, bool all_keypoints) {
  const auto cams = data_lines(dir / "cameras.txt");
  EXPECT(cams.size() == p.num_intrinsics_blocks());
  {
    std::istringstream s(cams.at(0));
    int id, w, h;
    std::string model;
    s >> id >> model >> w >> h;
    EXPECT(id == 1 && model == "PINHOLE" && w == 636 && h == 486);  // 2 cx, 2 cy rounded up
    for (std::size_t a = 0; a < p.intrinsics.size(); ++a) {
      double v;
      s >> v;
      EXPECT(v == p.intrinsics[a]);
    }
  }
  const auto im = data_lines(dir / "images.txt");
  EXPECT(im.size() == 2 * p.num_images());
  std::map<int, Image2D> images;
  for (std::size_t l = 0; l + 1 < im.size(); l += 2) {
    std::istringstream s(im[l]);
    int id;
    Image2D img;
    img.pose.resize(7);
    s >> id;
    for (double& v : img.pose) s >> v;
    s >> img.camera >> img.name;
    std::istringstream t(im[l + 1]);
    double x, y;
    long j;
    while (t >> x >> y >> j) img.pts.push_back({x, y, static_cast<double>(j)});
    images[id] = img;
  }
  for (std::size_t i = 0; i < p.num_images(); ++i) {
    const auto& img = images.at(static_cast<int>(i) + 1);
    for (int a = 0; a < 7; ++a) EXPECT(img.pose[static_cast<std::size_t>(a)] == p.poses[7 * i + static_cast<std::size_t>(a)]);
    EXPECT(img.camera == 1);
  }
  const auto pts = data_lines(dir / "points3D.txt");
  EXPECT(pts.size() == p.num_points());
  std::size_t track_entries = 0, unobserved = 0;
  for (const auto& [id, img] : images)
    for (const auto& q : img.pts) unobserved += q[2] < 0;
  for (const auto& line : pts) {
    std::istringstream s(line);
    int id, r, g, b;
    double X[3], err;
    s >> id >> X[0] >> X[1] >> X[2] >> r >> g >> b >> err;
    const auto j = static_cast<std::size_t>(id - 1);
    for (int a = 0; a < 3; ++a) EXPECT(X[a] == p.points[3 * j + static_cast<std::size_t>(a)]);
    EXPECT(err >= 0 && err < 5);
    int image, idx;
    while (s >> image >> idx) {
      const auto& e = images.at(image).pts.at(static_cast<std::size_t>(idx));
      EXPECT(static_cast<int>(e[2]) == id);  // the track points at a 2-D point of this 3-D point
      ++track_entries;
    }
  }
  EXPECT(track_entries == p.num_observations());
  EXPECT(all_keypoints ? unobserved > 0 : unobserved == 0);
}

}  // namespace

int main() {
  auto p = jba_test::random_problem(jba::CameraModel::Pinhole, 6, 80, 51);
  jba::solve(p);
  const fs::path dir = fs::temp_directory_path() / ("jba_export_" + std::to_string(std::random_device{}()));
  fs::create_directories(dir);

  // Observations only.
  jba::export_colmap(p, dir.string());
  check_model(p, dir, false);

  // Every keypoint: some keypoints observe no point.
  jba::Keypoints kp;
  std::vector<std::int32_t> obs_keypoint(p.num_observations());
  kp.begin = {0};
  for (std::size_t i = 0; i < p.num_images(); ++i) {
    for (std::size_t k = 0; k < p.num_observations(); ++k) {
      if (static_cast<std::size_t>(p.obs_image[k]) != i) continue;
      obs_keypoint[k] = static_cast<std::int32_t>((kp.xy.size() / 2) - static_cast<std::size_t>(kp.begin.back()));
      kp.xy.insert(kp.xy.end(), {p.obs_xy[2 * k], p.obs_xy[2 * k + 1]});
    }
    kp.xy.insert(kp.xy.end(), {1.5, 2.5, 3.5, 4.5});  // two unmatched keypoints
    kp.begin.push_back(static_cast<std::int64_t>(kp.xy.size() / 2));
  }
  jba::ColmapExportOptions o;
  o.keypoints = &kp;
  o.obs_keypoint = obs_keypoint;
  jba::export_colmap(p, dir.string(), o);
  check_model(p, dir, true);

  // Own format with results: read back, replaced on a second write.
  jba::CovarianceOptions co;
  co.points = {1, 2};
  const auto cov = jba::covariance(p, {}, co);
  const auto rep = jba::evaluate(p, {});
  const auto path = (dir / "result.jba").string();
  jba::write_result(path, p, &cov, &rep, &obs_keypoint);
  jba::write_result(path, jba::read_problem(path), &cov, &rep, &obs_keypoint);
  const auto back = jba::read_problem(path);
  std::map<std::string, const jba::RawSection*> by_name;
  for (const auto& s : back.extra) by_name[s.name] = &s;
  EXPECT(back.extra.size() == by_name.size());  // no duplicates after the second write
  EXPECT(by_name.count("result.pose_covariance") && by_name["result.pose_covariance"]->shape ==
                                                         (std::vector<std::uint64_t>{p.num_images(), 6, 6}));
  EXPECT(by_name.count("result.point_covariance") && by_name["result.point_covariance"]->shape ==
                                                          (std::vector<std::uint64_t>{2, 3, 3}));
  EXPECT(by_name.count("result.reprojection_error") && by_name.count("result.cost_by_type") &&
         by_name.count("result.obs_keypoint"));
  EXPECT(back.poses == p.poses && back.points == p.points);
  fs::remove_all(dir);
  return TEST_MAIN_RESULT();
}
