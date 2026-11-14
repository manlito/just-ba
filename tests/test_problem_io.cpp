// Problem file format: lossless round trip, preserved unknown sections, and clear
// errors on damaged files.

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "check.h"
#include "jba/problem_io.h"
#include "synthetic.h"

namespace fs = std::filesystem;

namespace {

std::string temp_path(const std::string& name) {
  return (fs::temp_directory_path() / ("jba_test_" + std::to_string(std::random_device{}()) + "_" + name)).string();
}

std::vector<char> slurp(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return std::vector<char>(std::istreambuf_iterator<char>(in), {});
}

void spit(const std::string& path, const std::vector<char>& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

bool same(const jba::Problem& a, const jba::Problem& b) {
  bool extra_same = a.extra.size() == b.extra.size();
  for (std::size_t i = 0; extra_same && i < a.extra.size(); ++i) {
    extra_same = a.extra[i].name == b.extra[i].name && a.extra[i].dtype == b.extra[i].dtype &&
                 a.extra[i].shape == b.extra[i].shape && a.extra[i].bytes == b.extra[i].bytes;
  }
  return a.model == b.model && a.intrinsics == b.intrinsics && a.intrinsics_free == b.intrinsics_free &&
         a.poses == b.poses && a.image_intrinsics == b.image_intrinsics && a.image_fixed == b.image_fixed &&
         a.points == b.points && a.point_fixed == b.point_fixed && a.obs_image == b.obs_image &&
         a.obs_point == b.obs_point && a.obs_xy == b.obs_xy && a.obs_sigma == b.obs_sigma && extra_same;
}

}  // namespace

int main() {
  // Minimal problem: required sections only.
  {
    const auto p = jba_test::random_problem(jba::CameraModel::SimpleRadial, 4, 30, 7);
    const auto path = temp_path("minimal.jba");
    jba::write_problem(path, p);
    const auto q = jba::read_problem(path);
    EXPECT(same(p, q));
    EXPECT(q.num_observations() == p.num_observations());
    EXPECT(slurp(path).size() % 8 == 0);
    fs::remove(path);
  }

  // Every optional section, several intrinsics blocks, and an unknown section.
  {
    auto p = jba_test::random_problem(jba::CameraModel::Radial, 3, 20, 8);
    const auto block = p.intrinsics;
    p.intrinsics.insert(p.intrinsics.end(), block.begin(), block.end());
    p.intrinsics_free = {jba::kFocal, jba::kFocal | jba::kDistortion};
    p.image_intrinsics = {0, 1, 1};
    p.image_fixed = {1, 0, 0};
    p.point_fixed.assign(p.num_points(), 0);
    p.point_fixed[3] = 1;
    p.obs_sigma.assign(p.num_observations(), 1.5);
    jba::RawSection future;
    future.name = "prior.relative.pairs";
    future.dtype = 2;  // int32
    future.shape = {2, 2};
    future.bytes.assign(16, 0);
    future.bytes[0] = 1;
    p.extra.push_back(future);

    const auto path = temp_path("full.jba");
    jba::write_problem(path, p);
    const auto q = jba::read_problem(path);
    EXPECT(same(p, q));

    // Writing what was read gives the same bytes.
    const auto path2 = temp_path("full2.jba");
    jba::write_problem(path2, q);
    EXPECT(slurp(path) == slurp(path2));

    // Damaged files are refused with an exception, never read silently.
    const auto bytes = slurp(path);
    const auto bad = temp_path("bad.jba");
    for (std::size_t cut : {std::size_t(3), std::size_t(17), bytes.size() / 2, bytes.size() - 8}) {
      spit(bad, std::vector<char>(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(cut)));
      EXPECT_THROWS(jba::read_problem(bad));
    }
    auto wrong_magic = bytes;
    wrong_magic[1] = 'X';
    spit(bad, wrong_magic);
    EXPECT_THROWS(jba::read_problem(bad));
    auto wrong_version = bytes;
    wrong_version[8] = 9;
    spit(bad, wrong_version);
    EXPECT_THROWS(jba::read_problem(bad));
    auto trailing = bytes;
    trailing.insert(trailing.end(), 8, 0);
    spit(bad, trailing);
    EXPECT_THROWS(jba::read_problem(bad));
    for (const auto& f : {path, path2, bad}) fs::remove(f);
  }

  // Invalid problems are refused on write.
  {
    auto p = jba_test::random_problem(jba::CameraModel::Pinhole, 2, 5, 9);
    const auto path = temp_path("invalid.jba");
    auto bad_index = p;
    bad_index.obs_point[0] = 1000;
    EXPECT_THROWS(jba::write_problem(path, bad_index));
    auto bad_quat = p;
    bad_quat.poses[0] = 2;
    EXPECT_THROWS(jba::write_problem(path, bad_quat));
    auto two_blocks = p;
    two_blocks.intrinsics.insert(two_blocks.intrinsics.end(), p.intrinsics.begin(), p.intrinsics.end());
    EXPECT_THROWS(jba::write_problem(path, two_blocks));  // image_intrinsics missing
    EXPECT(!jba::validate(p).size());
    fs::remove(path);
  }

  return TEST_MAIN_RESULT();
}
