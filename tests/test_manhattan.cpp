// Manhattan prior: line assignment, frame estimation and the support test
// (document 02, section 5; document 05's required cases), on synthetic rooms.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "check.h"
#include "jba/geometry.h"
#include "jba/manhattan.h"
#include "scene.h"

namespace {

constexpr double kDeg = 3.141592653589793 / 180;

void axis_dir(const double* q, int k, double* d) {
  double e[3] = {0, 0, 0};
  e[k] = 1;
  jba::quat_rotate(q, e, d);
}

// For each true axis, the estimated axis along it (the frame is defined up to the
// symmetry of the cube), and the worst angle between them.
double match_axes(const double* truth, const double* est, int* map) {
  double worst = 0;
  for (int t = 0; t < 3; ++t) {
    double dt[3];
    axis_dir(truth, t, dt);
    double best = -1;
    for (int e = 0; e < 3; ++e) {
      double de[3];
      axis_dir(est, e, de);
      const double c = std::abs(dt[0] * de[0] + dt[1] * de[1] + dt[2] * de[2]);
      if (c > best) {
        best = c;
        map[t] = e;
      }
    }
    worst = std::max(worst, std::acos(std::min(1.0, best)));
  }
  return worst;
}

struct Counts {
  int correct = 0, wrong = 0, missed = 0, distractor_assigned = 0, distractors = 0;
};

Counts score(const jba_test::Scene& sc, const jba::ManhattanAssignment& a, const int* map) {
  Counts c;
  for (std::size_t k = 0; k < a.axis.size(); ++k) {
    const int t = sc.line_truth_axis[k];
    if (t < 0) {
      ++c.distractors;
      if (a.axis[k] >= 0) ++c.distractor_assigned;
    } else if (a.axis[k] < 0) {
      ++c.missed;
    } else if (a.axis[k] == map[t]) {
      ++c.correct;
    } else {
      ++c.wrong;
    }
  }
  return c;
}

jba_test::SceneOptions room(jba_test::LineMode mode) {
  jba_test::SceneOptions o;
  o.kind = jba_test::SceneKind::Room;
  o.images = 40;
  o.points = 3000;
  o.lines = mode;
  return o;
}

}  // namespace

int main() {
  jba::ManhattanOptions mo;

  // A room seen from the true poses: the frame is found, nearly every segment is
  // assigned to its axis, none to a wrong one.
  {
    const auto sc = jba_test::make_scene(room(jba_test::LineMode::Manhattan));
    jba::Problem p = sc.input.scene;
    p.poses = sc.truth.poses;
    const auto a = jba::assign_lines(p, mo);
    int map[3];
    const double err = match_axes(sc.truth_frame, a.frame.data(), map);
    const Counts c = score(sc, a, map);
    std::printf("room, true poses: %zu segments, frame off by %.3f deg, support %.2f, correct %d, wrong %d, missed %d\n",
                p.num_lines(), err / kDeg, a.support, c.correct, c.wrong, c.missed);
    // Missed segments are mostly ambiguous ones: a horizontal edge at eye height has a
    // nearly horizontal plane, which holds both horizontal axes.
    EXPECT(p.num_lines() > 800);
    EXPECT(a.accepted);
    EXPECT(err < 0.1 * kDeg);
    EXPECT(map[2] == 2);  // the vertical is axis 2
    EXPECT(c.wrong <= 0.002 * (c.correct + c.wrong + c.missed));
    EXPECT(c.correct > 0.9 * (c.correct + c.wrong + c.missed));
    EXPECT(a.support > 0.9);
  }

  // From drifting tracker poses (heading drift as in document 05): still accepted;
  // segments in images whose heading has drifted beyond the threshold stay unassigned
  // rather than wrongly assigned.
  {
    auto o = room(jba_test::LineMode::Manhattan);
    o.drift_heading_per_m = 0.03;
    const auto sc = jba_test::make_scene(o);
    const auto a = jba::assign_lines(sc.input.scene, mo);
    int map[3];
    const double err = match_axes(sc.truth_frame, a.frame.data(), map);
    const Counts c = score(sc, a, map);
    std::printf("room, drifting tracker: frame off by %.2f deg (after alignment of the start), support %.2f, "
                "correct %d, wrong %d, missed %d\n", err / kDeg, a.support, c.correct, c.wrong, c.missed);
    EXPECT(a.accepted);
    EXPECT(c.wrong < 0.01 * (c.correct + c.wrong));
  }

  // Mixed scene: off-axis segments (sloped, floating, a wing 30 degrees off) are left
  // out, and the prior is still accepted.
  {
    const auto sc = jba_test::make_scene(room(jba_test::LineMode::Mixed));
    jba::Problem p = sc.input.scene;
    p.poses = sc.truth.poses;
    const auto a = jba::assign_lines(p, mo);
    int map[3];
    const double err = match_axes(sc.truth_frame, a.frame.data(), map);
    const Counts c = score(sc, a, map);
    std::printf("mixed room: frame off by %.3f deg, support %.2f, correct %d, wrong %d, missed %d, "
                "distractors assigned %d of %d\n", err / kDeg, a.support, c.correct, c.wrong, c.missed,
                c.distractor_assigned, c.distractors);
    EXPECT(a.accepted);
    EXPECT(err < 0.2 * kDeg);
    EXPECT(c.wrong <= 0.002 * (c.correct + c.wrong + c.missed));
    EXPECT(c.distractor_assigned < 0.1 * c.distractors);
  }

  // Not Manhattan: segments in random directions. The support test refuses.
  {
    const auto sc = jba_test::make_scene(room(jba_test::LineMode::Random));
    jba::Problem p = sc.input.scene;
    p.poses = sc.truth.poses;
    const auto a = jba::assign_lines(p, mo);
    std::printf("random segments: support %.2f, accepted %d: %s\n", a.support, int(a.accepted), a.message.c_str());
    EXPECT(!a.accepted);
    EXPECT(!a.message.empty());
    EXPECT(a.support < 0.2);
    auto q = p;
    jba::apply_assignment(q, a, 1);
    EXPECT(!q.has_manhattan());
  }

  // Three degrees of freedom: a working frame that is not gravity-aligned. The
  // strongest direction among the segments is taken as axis 2.
  {
    const auto sc = jba_test::make_scene(room(jba_test::LineMode::Manhattan));
    jba::Problem p = sc.input.scene;
    p.poses = sc.truth.poses;
    // Rotate the world by W: R' = R W^T; the true frame becomes W R_M.
    const double w[3] = {0.4, -0.3, 0.9};
    double W[4], Wc[4];
    jba::quat_exp(w, W);
    Wc[0] = W[0], Wc[1] = -W[1], Wc[2] = -W[2], Wc[3] = -W[3];
    for (std::size_t i = 0; i < p.num_images(); ++i) {
      double q[4];
      jba::quat_multiply(&p.poses[7 * i], Wc, q);
      jba::quat_normalize(q);
      std::copy_n(q, 4, &p.poses[7 * i]);
    }
    double truth[4];
    jba::quat_multiply(W, sc.truth_frame, truth);
    auto o3 = mo;
    o3.dof = 3;
    const auto a = jba::assign_lines(p, o3);
    int map[3];
    const double err = match_axes(truth, a.frame.data(), map);
    const Counts c = score(sc, a, map);
    std::printf("three freedoms, rotated world: frame off by %.3f deg, correct %d, wrong %d, missed %d\n", err / kDeg,
                c.correct, c.wrong, c.missed);
    EXPECT(a.accepted);
    EXPECT(err < 0.1 * kDeg);
    EXPECT(c.wrong <= 0.002 * (c.correct + c.wrong + c.missed));
    // With one freedom the same rotated world is wrong about its vertical: refused or poor.
    const auto a1 = jba::assign_lines(p, mo);
    EXPECT(!a1.accepted || a1.support < a.support);
  }

  // A given frame is used as it is.
  {
    const auto sc = jba_test::make_scene(room(jba_test::LineMode::Manhattan));
    jba::Problem p = sc.input.scene;
    p.poses = sc.truth.poses;
    const auto a = jba::assign_lines(p, mo, sc.truth_frame);
    EXPECT(a.accepted);
    for (int c = 0; c < 4; ++c) EXPECT(a.frame[static_cast<std::size_t>(c)] == sc.truth_frame[c]);
    const int map[3] = {0, 1, 2};
    const Counts c = score(sc, a, map);
    EXPECT(c.wrong <= 0.002 * (c.correct + c.wrong + c.missed) && c.correct > 0.9 * (c.correct + c.wrong + c.missed));
    jba::apply_assignment(p, a, 1);
    EXPECT(p.has_manhattan() && jba::validate(p).empty());
  }
  return TEST_MAIN_RESULT();
}
