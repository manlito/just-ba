// The mapper's sessions (D152): a map grown one image at a time, each image with its
// matches to earlier ones, then finished from the live map with Mapper::run_from. On
// the generated scenes: every image placed as it comes, the live map near the truth and
// the batch run, the finish as good as the batch run; a session fed only sequential
// pairs; a window of every image; bad input; and run_from started from a batch result.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "compare.h"
#include "mapper.h"
#include "scenes.h"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  if (ok) return;
  std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  ++failures;
}

std::size_t count(const std::vector<char>& v) {
  std::size_t n = 0;
  for (char c : v) n += c != 0;
  return n;
}

std::vector<char> all(std::size_t n) { return std::vector<char>(n, 1); }

// The capture's images fed in order, each with the pairs whose later image it is (at
// most `seq` images back when seq > 0).
demo::Session feed(demo::Mapper& mapper, const demo::Scene& scene, const demo::SessionOptions& so, int seq,
                   std::vector<demo::SessionStep>* steps) {
  const demo::Capture& c = scene.capture;
  demo::Session s(mapper, c.model, c.intrinsics, !c.focal_unknown.empty() && c.focal_unknown[0],
                  static_cast<std::int32_t>(scene.width), static_cast<std::int32_t>(scene.height), so);
  for (std::size_t i = 0; i < c.num_images(); ++i) {
    demo::SessionImage im;
    im.keypoints = c.keypoints[i];
    for (demo::Pair p : c.pairs) {
      if (p.a > p.b) {
        std::swap(p.a, p.b);
        std::swap(p.ka, p.kb);
      }
      if (static_cast<std::size_t>(p.b) != i || (seq > 0 && p.b - p.a > seq)) continue;
      im.pairs.push_back(p);
    }
    steps->push_back(s.add(im));
  }
  return s;
}

void scene_case(const std::string& name) {
  const demo::Scene scene = demo::make_scene(name);
  demo::MapperOptions mo;
  demo::thresholds_for_noise(scene.noise_px, mo);
  demo::Mapper mapper(mo);
  const std::size_t n = scene.capture.num_images();
  const demo::Reconstruction batch = mapper.run(scene.capture);
  check(batch.ok() && count(batch.registered) == n, name + ": the batch run places every image");
  const std::vector<char> truth = all(n);

  for (const int window : {8, 3, 0}) {
    const std::string label = name + " window " + std::to_string(window);
    demo::SessionOptions so;
    so.window = window;
    std::vector<demo::SessionStep> steps;
    demo::Session s = feed(mapper, scene, so, 0, &steps);
    bool ok = true;
    int started = -1, late = 0;
    for (const demo::SessionStep& st : steps) {
      ok = ok && st.ok();
      if (st.started) started = st.image;
      if (started >= 0 && st.image > started &&
          std::find(st.placed.begin(), st.placed.end(), st.image) == st.placed.end())
        ++late;
    }
    check(ok, label + ": every step succeeds");
    // Pairs reaching 16 degrees start it at once; one floor lower per start_patience
    // images (the grid's pairs are narrow).
    check(started >= 0 && started <= 2 * so.start_patience, label + ": the map starts within the first images");
    check(late == 0, label + ": every image after the start is placed as it comes");
    const demo::Snapshot live = s.snapshot();
    check(count(*live.registered) == n, label + ": every image is in the live map");
    check(live.points.size() == 3 * live.point_track.size() && !live.points.empty(), label + ": points with tracks");
    // Within the window, and the held images outside it.
    const demo::SessionStep& final_step = steps.back();
    if (window > 0)
      check(static_cast<int>(final_step.window.size()) <= window, label + ": the window holds at most its size");
    if (window > 0 && n > static_cast<std::size_t>(window) + 2)
      check(!final_step.held.empty(), label + ": older images are held");
    if (window == 0) check(final_step.held.empty(), label + ": window 0 adjusts every image");
    // Landmarks: keypoints of the last image that observe a point of the map.
    const std::vector<std::int32_t> kp = s.keypoint_points(n - 1);
    std::size_t landmarks = 0;
    bool known = true;
    for (std::int32_t t : kp) {
      if (t < 0) continue;
      ++landmarks;
      known = known && std::binary_search(live.point_track.begin(), live.point_track.end(), t);
    }
    check(kp.size() == scene.capture.keypoints[n - 1].size() / 2, label + ": one value per keypoint");
    check(landmarks >= mo.min_pnp_inliers && known, label + ": the last image's landmarks are points of the map");

    const demo::PoseAgreement vs_truth = demo::compare_poses(*live.registered, *live.poses, truth, scene.poses);
    check(vs_truth.rot_median_deg < 0.5 && vs_truth.centre_median_pct < 1.0,
          label + ": the live map is near the truth (" + std::to_string(vs_truth.rot_median_deg) + "°, " +
              std::to_string(vs_truth.centre_median_pct) + " %)");

    const demo::Reconstruction fin = s.finish(scene.capture);
    check(fin.ok() && count(fin.registered) == n, label + ": the finish places every image");
    const demo::PoseAgreement vs_batch = demo::compare_poses(fin.registered, fin.poses, batch.registered, batch.poses);
    check(vs_batch.centre_median_pct < 0.1 && vs_batch.rot_median_deg < 0.05,
          label + ": the finish agrees with the batch run (" + std::to_string(vs_batch.centre_median_pct) + " %)");
    check(fin.mean_error_px <= 1.05 * batch.mean_error_px, label + ": the finish's error is no worse than batch's");
    check(fin.first_a < 0 && !fin.all_models.empty() && fin.all_models[0].first_pair < 0,
          label + ": the finish's first model is the live map's");
    std::printf("%-16s map from image %d; live vs truth %.3f°, %.3f %%; finish vs batch %.4f %%, %.3f px (batch %.3f)\n",
                label.c_str(), started, vs_truth.rot_median_deg, vs_truth.centre_median_pct,
                vs_batch.centre_median_pct, fin.mean_error_px, batch.mean_error_px);
  }

  // Sequential pairs only, three back: as a capture's live matching gives them.
  {
    std::vector<demo::SessionStep> steps;
    demo::Session s = feed(mapper, scene, {}, 3, &steps);
    const demo::Snapshot live = s.snapshot();
    const demo::PoseAgreement a = demo::compare_poses(*live.registered, *live.poses, truth, scene.poses);
    check(count(*live.registered) == n, name + " sequential: every image is in the live map");
    check(a.centre_median_pct < 1.0, name + " sequential: the live map is near the truth");
    std::printf("%-16s sequential: %zu of %zu live, %.3f %% from the truth\n", name.c_str(), count(*live.registered), n,
                a.centre_median_pct);
  }

  // run_from a batch result: the same images, the same poses to rounding.
  {
    demo::StartFrom start;
    start.registered = batch.registered;
    start.poses = batch.poses;
    start.intrinsics = batch.problem.intrinsics;
    const demo::Reconstruction r = mapper.run_from(scene.capture, start);
    const demo::PoseAgreement a = demo::compare_poses(r.registered, r.poses, batch.registered, batch.poses);
    check(r.ok() && count(r.registered) == n, name + " run_from: every image");
    check(a.centre_median_pct < 0.05, name + " run_from: the batch result stays where it was");
    check(r.start_left_out == 0, name + " run_from: a good start keeps every image");

    // One start image turned 30 degrees: it disagrees with its pairs, is left out of
    // the start and placed again, where the others put it.
    demo::StartFrom bent = start;
    const std::size_t wrong = n / 2;
    double* q = &bent.poses[7 * wrong];
    const double h = 0.2588190451;  // sin(15°): a 30° turn about the camera's y axis
    const double w0 = q[0], x0 = q[1], y0 = q[2], z0 = q[3];
    const double c = 0.9659258263;
    q[0] = c * w0 - h * y0;
    q[1] = c * x0 + h * z0;
    q[2] = c * y0 + h * w0;
    q[3] = c * z0 - h * x0;
    const demo::Reconstruction rb = mapper.run_from(scene.capture, bent);
    const demo::PoseAgreement ab = demo::compare_poses(rb.registered, rb.poses, batch.registered, batch.poses);
    check(rb.ok() && count(rb.registered) == n && rb.start_left_out >= 1,
          name + " run_from: a wrong start image is left out (" + std::to_string(rb.start_left_out) + ")");
    check(ab.rot_max_deg < 0.5, name + " run_from: and placed where it belongs (" + std::to_string(ab.rot_max_deg) + "°)");
  }
}

}  // namespace

int main() {
  for (const std::string& name : demo::scene_names()) scene_case(name);

  // A capture that loses its way: the orbit's two halves share no pair. The first half
  // is one piece; images of the second can't be placed in it, so a second piece starts
  // from them; the finish starts from the larger and grows the other as another model.
  {
    const demo::Scene scene = demo::make_scene("orbit");
    demo::MapperOptions mo;
    demo::thresholds_for_noise(scene.noise_px, mo);
    demo::Mapper mapper(mo);
    demo::Scene cut = scene;
    const std::int32_t half = static_cast<std::int32_t>(scene.capture.num_images() / 2);
    cut.capture.pairs.clear();
    for (const demo::Pair& p : scene.capture.pairs)
      if ((p.a < half) == (p.b < half)) cut.capture.pairs.push_back(p);
    std::vector<demo::SessionStep> steps;
    demo::Session s = feed(mapper, cut, {}, 0, &steps);
    check(s.num_pieces() == 2, "cut: two pieces (" + std::to_string(s.num_pieces()) + ")");
    const demo::Snapshot first = s.piece(0), now = s.snapshot();
    std::size_t in_first = 0, in_second = 0, both = 0;
    for (std::size_t i = 0; i < first.registered->size(); ++i) in_first += (*first.registered)[i] != 0;
    for (std::size_t i = 0; i < now.registered->size(); ++i) {
      in_second += (*now.registered)[i] != 0;
      both += (*now.registered)[i] && i < first.registered->size() && (*first.registered)[i];
    }
    check(in_first == static_cast<std::size_t>(half), "cut: the first piece is the first half");
    check(in_second + 3 >= static_cast<std::size_t>(half) && both == 0,
          "cut: the second piece is the second half (" + std::to_string(in_second) + ")");
    int restarted_at = -1;
    for (const demo::SessionStep& st : steps)
      if (st.started && st.piece == 1) restarted_at = st.image;
    check(restarted_at > half && restarted_at <= half + 4, "cut: the second piece starts soon after the cut");
    const demo::Reconstruction fin = s.finish(cut.capture);
    check(fin.ok() && fin.models == 2, "cut: the finish grows the other half as another model");
    std::printf("cut orbit: pieces of %zu and %zu images, the second from image %d; finish %d models\n", in_first,
                in_second, restarted_at, fin.models);
  }

  // Bad input: a pair naming a later image, or a keypoint that does not exist, is left
  // out and said so; the image still joins, so the caller's numbering holds.
  {
    const demo::Scene scene = demo::make_scene("orbit");
    demo::MapperOptions mo;
    demo::thresholds_for_noise(scene.noise_px, mo);
    demo::Mapper mapper(mo);
    const demo::Capture& c = scene.capture;
    demo::Session s(mapper, c.model, c.intrinsics, false, 0, 0);
    demo::SessionImage im;
    im.keypoints = c.keypoints[0];
    demo::Pair p;
    p.a = 1;  // not added yet
    p.ka = {0};
    p.kb = {0};
    im.pairs = {p};
    demo::SessionStep st = s.add(im);
    check(!st.ok() && s.num_images() == 1 && st.image == 0, "bad input: a pair to a later image is refused");
    im.keypoints = c.keypoints[1];
    p.a = 0;
    p.ka = {static_cast<std::int32_t>(c.keypoints[0].size())};  // past the end
    im.pairs = {p};
    st = s.add(im);
    check(!st.ok() && s.num_images() == 2 && s.capture().pairs.empty(), "bad input: a keypoint past the end");
    check(!s.started(), "bad input: no map from nothing");
    const demo::Reconstruction r = s.finish(s.capture());
    check(!r.ok(), "bad input: finishing without pairs fails as run does");
    bool threw = false;
    try {
      demo::Session wrong(mapper, c.model, {1.0}, false, 0, 0);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    check(threw, "bad input: intrinsics of another model");
  }

  if (failures) std::fprintf(stderr, "%d check(s) failed\n", failures);
  else std::printf("all session checks passed\n");
  return failures ? 1 : 0;
}
