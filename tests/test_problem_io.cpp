// Problem file format: lossless round trip, preserved unknown sections, and clear
// errors on damaged files.

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "jba/priors.h"
#include "jba/problem_io.h"
#include "jba/solver.h"
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

// Equal, NaN equal to NaN.
bool same_nan(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (!(a[i] == b[i] || (std::isnan(a[i]) && std::isnan(b[i])))) return false;
  return true;
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
         a.obs_point == b.obs_point && a.obs_xy == b.obs_xy && a.obs_sigma == b.obs_sigma &&
         a.rel_images == b.rel_images && a.rel_pose == b.rel_pose && a.rel_sigma == b.rel_sigma &&
         a.rel_sqrt_info == b.rel_sqrt_info && a.glob_image == b.glob_image && a.glob_set == b.glob_set &&
         a.glob_pose == b.glob_pose && a.glob_sigma == b.glob_sigma && a.glob_sqrt_info == b.glob_sqrt_info &&
         a.set_relation == b.set_relation && a.set_transform == b.set_transform && a.loss_type == b.loss_type &&
         a.loss_scale == b.loss_scale && a.gravity_world == b.gravity_world && a.grav_image == b.grav_image &&
         a.grav_dir == b.grav_dir && a.grav_sigma == b.grav_sigma && a.line_image == b.line_image &&
         a.line_xy == b.line_xy && a.line_sigma == b.line_sigma && a.line_axis == b.line_axis &&
         a.manhattan_frame == b.manhattan_frame && a.manhattan_dof == b.manhattan_dof &&
         same_nan(a.obs_depth, b.obs_depth) && same_nan(a.obs_depth_sigma, b.obs_depth_sigma) &&
         a.depth_kind == b.depth_kind && a.depth_form == b.depth_form && a.depth_correction == b.depth_correction &&
         a.depth_correction_free == b.depth_correction_free && extra_same;
}

}  // namespace

int main() {
  // Pose priors: every prior section, both weight forms, several sets.
  {
    auto p = jba_test::random_problem(jba::CameraModel::Pinhole, 4, 20, 9);
    jba::add_relative_priors_from_poses(p, p.poses, {0, 1, 1, 2, 2, 3});
    p.rel_sqrt_info.assign(36 * p.num_relative_priors(), 0.0);
    for (std::size_t k = 0; k < p.num_relative_priors(); ++k)
      for (int a = 0; a < 6; ++a) p.rel_sqrt_info[36 * k + 7 * static_cast<std::size_t>(a)] = 10.0 + a;
    p.glob_image = {0, 2, 3};
    p.glob_set = {0, 1, 1};
    p.glob_pose.assign(p.poses.begin(), p.poses.begin() + 21);
    p.glob_sigma = {0.01, 0.1, 0.02, std::numeric_limits<double>::infinity(), 0.03, 0.3};
    p.set_relation = {jba::FrameRelation::Identity, jba::FrameRelation::Similarity};
    p.set_transform = {1, 0, 0, 0, 0, 0, 0, 1, 0.6, 0.8, 0, 0, 1, 2, 3, 1.5};
    p.gravity_world = {0, 0.6, -0.8};
    p.grav_image = {1, 3};
    p.grav_dir = {0, 1, 0, 0.6, 0, 0.8};
    p.grav_sigma = {0.01, 0.02};
    p.line_image = {0, 2, 3};
    p.line_xy = {10, 20, 200, 25, 30, 40, 35, 300, 1, 2, 3, 4};
    p.line_sigma = {0.5, 1, 2};
    p.line_axis = {0, 2, -1};
    p.manhattan_frame = {0.6, 0.8, 0, 0};  // any rotation, with three freedoms
    p.manhattan_dof = 3;
    p.obs_depth.assign(p.num_observations(), std::nan(""));
    p.obs_depth[0] = 2.5;
    p.obs_depth[3] = 4.0;
    p.obs_depth_sigma.assign(p.num_observations(), std::nan(""));
    p.obs_depth_sigma[0] = 0.05;
    p.obs_depth_sigma[3] = 0.1;
    p.depth_kind = jba::DepthKind::Range;
    p.depth_form = jba::DepthForm::Inverse;
    p.depth_correction = {1, 0, 1.1, 0.02, 0.9, -0.1, 1, 0};
    p.depth_correction_free = {0, 3, 1, 2};
    const auto path = temp_path("priors.jba");
    jba::write_problem(path, p);
    const auto q = jba::read_problem(path);
    EXPECT(same(p, q));
    EXPECT(q.num_relative_priors() == 3 && q.num_global_priors() == 3 && q.num_prior_sets() == 2);
    fs::remove(path);

    // Invalid priors are refused.
    auto bad = p;
    bad.rel_images[1] = 0;  // ties image 0 to itself
    EXPECT(!jba::validate(bad).empty());
    bad = p;
    bad.rel_sigma = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};  // both weight forms
    EXPECT(!jba::validate(bad).empty());
    bad = p;
    bad.glob_set[1] = 5;
    EXPECT(!jba::validate(bad).empty());
    bad = p;
    bad.set_transform[4] = 0.5;  // identity set with a non-identity transform
    EXPECT(!jba::validate(bad).empty());
    bad = p;
    bad.glob_sigma[0] = 0;
    EXPECT(!jba::validate(bad).empty());
    bad = p;
    bad.grav_dir[0] = 0.1;  // not unit
    EXPECT(!jba::validate(bad).empty());
    bad = p;
    bad.grav_image[1] = 4;
    EXPECT(!jba::validate(bad).empty());
    bad = p;
    bad.gravity_world = {0, 1};
    EXPECT(!jba::validate(bad).empty());
    bad = p;
    bad.grav_sigma = {0.01};
    EXPECT(!jba::validate(bad).empty());
    bad = p;
    bad.line_axis[1] = 3;
    EXPECT(!jba::validate(bad).empty());
    bad = p;
    bad.line_xy[4] = bad.line_xy[6];
    bad.line_xy[5] = bad.line_xy[7];  // zero length
    EXPECT(!jba::validate(bad).empty());
    bad = p;
    bad.obs_depth[3] = -1;
    EXPECT(!jba::validate(bad).empty());
    bad = p;
    bad.obs_depth_sigma[3] = 0;
    EXPECT(!jba::validate(bad).empty());
    bad = p;
    bad.depth_correction[2] = 0;  // non-positive scale
    EXPECT(!jba::validate(bad).empty());
    bad = p;
    bad.depth_correction_free[1] = 4;
    EXPECT(!jba::validate(bad).empty());
    bad = p;
    bad.manhattan_dof = 2;
    EXPECT(!jba::validate(bad).empty());
    // With one freedom, axis 2 must be up. A rotation of 90 degrees about x takes it to
    // (0, -1, 0), which is up for gravity (0, 1, 0) but not for this problem's.
    bad = p;
    bad.manhattan_dof = 1;
    EXPECT(!jba::validate(bad).empty());
    bad.manhattan_frame = {std::sqrt(0.5), std::sqrt(0.5), 0, 0};
    EXPECT(!jba::validate(bad).empty());
    bad.gravity_world = {0, 1, 0};
    EXPECT(jba::validate(bad).empty());
  }

  // Losses stored with the problem (loss.* sections, D80): round trip, the options
  // they produce, and validation.
  {
    auto p = jba_test::random_problem(jba::CameraModel::Pinhole, 4, 20, 3);
    jba::SolverOptions o;
    o.loss = {jba::LossType::Cauchy, 2.0};
    o.relative_prior_loss = {jba::LossType::Huber, 4.0};
    o.global_prior_loss = {jba::LossType::None, 1.0};
    jba::store_losses(p, o);
    EXPECT(p.loss_type.size() == jba::kNumLossTypes && p.loss_scale[0] == 2.0);
    const auto path = temp_path("losses.jba");
    jba::write_problem(path, p);
    const auto q = jba::read_problem(path);
    EXPECT(same(p, q) && q.extra.empty());
    fs::remove(path);
    const auto back = jba::with_problem_losses(jba::SolverOptions{}, q);
    EXPECT(back.loss.type == jba::LossType::Cauchy && back.loss.scale == 2.0);
    EXPECT(back.relative_prior_loss.type == jba::LossType::Huber && back.relative_prior_loss.scale == 4.0);
    EXPECT(back.global_prior_loss.type == jba::LossType::None);
    // Fewer entries than residual types: the rest keep the options' values.
    auto one = q;
    one.loss_type.resize(1);
    one.loss_scale.resize(1);
    const auto partial = jba::with_problem_losses(jba::SolverOptions{}, one);
    EXPECT(partial.loss.type == jba::LossType::Cauchy &&
           partial.relative_prior_loss.type == jba::SolverOptions{}.relative_prior_loss.type);
    // No sections: the options are unchanged.
    EXPECT(jba::with_problem_losses(o, jba::Problem{}).loss.scale == 2.0);

    auto bad = q;
    bad.loss_scale.pop_back();
    EXPECT(!jba::validate(bad).empty());
    bad = q;
    bad.loss_type[1] = 9;
    EXPECT(!jba::validate(bad).empty());
    bad = q;
    bad.loss_scale[0] = 0;
    EXPECT(!jba::validate(bad).empty());
    bad = q;
    bad.loss_type.push_back(0);
    bad.loss_scale.push_back(1);
    EXPECT(!jba::validate(bad).empty());  // more than there are residual types
  }

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
    future.name = "prior.relative.pairs";  // not a section this version reads
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
    // A section name may appear once: a second copy is refused by the reader, and
    // the writer refuses to produce one.
    auto twice = bytes;
    twice.insert(twice.end(), bytes.end() - 64, bytes.end());  // the unknown section again: 48 header, 16 data
    ++twice[12];                                               // section count
    spit(bad, twice);
    std::string message;
    try {
      jba::read_problem(bad);
    } catch (const std::exception& e) {
      message = e.what();
    }
    EXPECT(message.find("'prior.relative.pairs' appears more than once") != std::string::npos);
    auto repeated = p;
    repeated.extra.push_back(future);
    EXPECT_THROWS(jba::write_problem(bad, repeated));
    auto shadow = p;
    shadow.extra[0].name = "obs.xy";
    EXPECT_THROWS(jba::write_problem(bad, shadow));
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
    // Values that are not finite are named, not left to fail the solve later.
    for (auto member : {&jba::Problem::intrinsics, &jba::Problem::poses, &jba::Problem::points, &jba::Problem::obs_xy}) {
      auto bad = p;
      (bad.*member).back() = std::numeric_limits<double>::infinity();
      EXPECT(jba::validate(bad).find("not finite") != std::string::npos);
    }
    auto bad_sigma = p;
    bad_sigma.obs_sigma.assign(p.num_observations(), std::numeric_limits<double>::infinity());
    EXPECT(!jba::validate(bad_sigma).empty());
    fs::remove(path);
  }

  return TEST_MAIN_RESULT();
}
