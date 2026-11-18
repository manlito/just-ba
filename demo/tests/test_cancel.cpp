// The mapper's progress callback (D146): an always-continue callback leaves the result
// bit-identical to a run without one, and a callback that returns false stops the run,
// which comes back cancelled, without calling it again. On the generated orbit, as
// given and with its focal length off by 2.2 times (the focal search's stage).
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "mapper.h"
#include "scenes.h"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  if (ok) return;
  std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  ++failures;
}

template <class T>
bool same(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

// The snapshots on_step replays after the run.
struct Steps {
  std::vector<std::string> stages;
  std::vector<std::vector<double>> points;
};

demo::StepCallback recorder(Steps& s) {
  return [&s](const demo::Snapshot& snap) {
    s.stages.push_back(snap.stage);
    s.points.push_back(snap.points);
  };
}

void compare(const demo::Reconstruction& a, const Steps& sa, const demo::Reconstruction& b, const Steps& sb,
             const std::string& label) {
  check(a.ok() && b.ok(), label + ": both runs succeed");
  check(!a.cancelled && !b.cancelled, label + ": neither run is cancelled");
  check(same(a.registered, b.registered), label + ": registered images");
  check(same(a.poses, b.poses), label + ": poses, bit for bit");
  check(same(a.problem.points, b.problem.points), label + ": points, bit for bit");
  check(same(a.problem.intrinsics, b.problem.intrinsics), label + ": intrinsics, bit for bit");
  check(same(a.point_track, b.point_track), label + ": point tracks");
  check(std::memcmp(&a.mean_error_px, &b.mean_error_px, sizeof(double)) == 0, label + ": mean error, bit for bit");
  check(a.focal_factor == b.focal_factor, label + ": focal factor");
  check(a.adjustments == b.adjustments && a.models == b.models, label + ": adjustments and models");
  check(sa.stages == sb.stages, label + ": on_step's stages");
  bool points_same = sa.points.size() == sb.points.size();
  for (std::size_t k = 0; points_same && k < sa.points.size(); ++k) points_same = same(sa.points[k], sb.points[k]);
  check(points_same, label + ": on_step's points, bit for bit");
}

void run_case(const std::string& label, const demo::Capture& capture, const demo::MapperOptions& mo) {
  // Without a callback.
  Steps s0;
  const demo::Reconstruction r0 = demo::Mapper(mo).run(capture, recorder(s0));
  check(r0.ok(), label + ": the run without a callback succeeds (" + r0.error + ")");

  // Always continue: the same result; every stage seen, in order, live where promised.
  Steps s1;
  std::vector<std::string> stages;
  int calls = 0, live_ok = 1, iterations = 0;
  const demo::Reconstruction r1 = demo::Mapper(mo).run(capture, recorder(s1), [&](const demo::Progress& p) {
    ++calls;
    if (stages.empty() || stages.back() != p.stage) stages.push_back(p.stage);
    const bool step = p.stage == "first pair" || p.stage == "register" || p.stage == "final";
    if (step != (p.live != nullptr)) live_ok = 0;
    if (p.live && (p.live->stage != p.stage || p.live->poses == nullptr || p.live->registered == nullptr)) live_ok = 0;
    if (p.stage == "adjust") ++iterations;
    if ((p.stage == "adjust") != (p.iteration >= 0)) live_ok = 0;
    if (p.images != static_cast<int>(capture.num_images())) live_ok = 0;
    return true;
  });
  compare(r0, s0, r1, s1, label + " (always continue)");
  check(live_ok == 1, label + ": live snapshots on the steps only, iterations on \"adjust\" only");
  check(iterations > r1.adjustments, label + ": solver iterations reported");
  const std::set<std::string> seen(stages.begin(), stages.end());
  for (const char* st : {"verify", "tracks", "first pair", "register", "final", "adjust"})
    check(seen.count(st) == 1, label + ": stage \"" + st + "\" reported");
  check(mo.search_focal == (seen.count("focal search") == 1), label + ": \"focal search\" reported when searched");
  check(!stages.empty() && stages.front() == (mo.search_focal ? "focal search" : "verify"), label + ": first stage");
  check(stages.back() == "final", label + ": last stage is \"final\"");
  std::printf("%s: %d progress calls, %d solver iterations, %d adjustments\n", label.c_str(), calls, iterations,
              r1.adjustments);

  // Stop after n calls: cancelled, the callback not called again, no replay of on_step.
  for (const int n : {1, 2, 3, calls / 4, calls / 2, calls - 1, calls}) {
    if (n < 1) continue;
    int count = 0, steps = 0;
    const demo::Reconstruction r = demo::Mapper(mo).run(
        capture, [&](const demo::Snapshot&) { ++steps; }, [&](const demo::Progress&) { return ++count < n; });
    const std::string at = label + ", stopped at call " + std::to_string(n);
    check(r.cancelled, at + ": cancelled");
    check(r.error == "cancelled", at + ": error is \"cancelled\" (" + r.error + ")");
    check(!r.ok(), at + ": not ok");
    check(count == n, at + ": no call after the one that stopped (" + std::to_string(count) + ")");
    check(steps == 0, at + ": on_step not replayed");
  }
}

}  // namespace

int main() {
  const demo::Scene scene = demo::make_scene("orbit");
  demo::MapperOptions mo;
  demo::thresholds_for_noise(scene.noise_px, mo);
  run_case("orbit", scene.capture, mo);

  // Focal length 2.2 times the truth, marked as a guess: searched for, then refined.
  demo::Capture guessed = scene.capture;
  const std::size_t np = static_cast<std::size_t>(jba::num_params(guessed.model));
  const int n_focal = guessed.model == jba::CameraModel::Pinhole ? 2 : 1;
  for (std::size_t b = 0; b < guessed.intrinsics.size() / np; ++b)
    for (int k = 0; k < n_focal; ++k) guessed.intrinsics[np * b + static_cast<std::size_t>(k)] *= 2.2;
  guessed.focal_unknown.assign(guessed.intrinsics.size() / np, 1);
  demo::MapperOptions mf = mo;
  mf.search_focal = true;
  mf.refine_intrinsics = jba::kFocal | jba::kDistortion;
  run_case("orbit, focal searched", guessed, mf);

  if (failures) std::fprintf(stderr, "%d checks failed\n", failures);
  else std::printf("all checks passed\n");
  return failures ? 1 : 0;
}
