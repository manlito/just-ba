// jba_demo: the demo app's command line.
//
//   jba_demo synth [--scene orbit|wall|grid|all] [--seed n] [--noise px] [--wrong f] [--false-pairs n]
//                  [--write dir] [common options]
//     Reconstructs the generated scenes and compares each with its ground truth.
//     --write: also writes each scene as a capture file (<dir>/<scene>.jbc), truth as reference.
//
//   jba_demo run capture.jbc [--intrinsics capture|reference] [--thresholds colmap|demo] [--check]
//                [common options]
//     Reconstructs a capture file (from demo/tools/colmap_capture.py) and, when it holds a
//     reference reconstruction, compares with it (document 05, level 3).
//
//   common options: [--poses auto|cpu|metal|cuda|webgpu] [--ba cpu|metal|cuda|webgpu] [--threads n]
//                   [--export dir] [-v]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

#include "capture_io.h"
#include "compare.h"
#include "evaluate.h"
#include "jba/export.h"
#include "mapper.h"
#include "scenes.h"

namespace {

int usage() {
  std::fprintf(stderr,
               "usage: jba_demo synth [--scene orbit|wall|grid|all] [--seed n] [--noise px] [--wrong f]\n"
               "                      [--false-pairs n] [--write dir] [options]\n"
               "       jba_demo run capture.jbc [--intrinsics capture|reference] [--thresholds colmap|demo]\n"
               "                    [--check] [options]\n"
               "options: [--poses auto|cpu|metal|cuda|webgpu] [--ba cpu|metal|cuda|webgpu] [--threads n]\n"
               "         [--export dir] [-v]\n");
  return 2;
}

bool pose_backend_from_name(const std::string& s, std::int32_t* b) {
  if (s == "auto") *b = JPO_BACKEND_AUTO;
  else if (s == "cpu") *b = JPO_BACKEND_CPU;
  else if (s == "metal") *b = JPO_BACKEND_METAL;
  else if (s == "cuda") *b = JPO_BACKEND_CUDA;
  else if (s == "webgpu") *b = JPO_BACKEND_WEBGPU;
  else return false;
  return true;
}

struct Common {
  demo::MapperOptions mapper;
  std::string export_dir;
  bool verbose = false;
};

// Parses the options both commands take; false for one it does not know.
bool common_option(const std::vector<std::string>& args, std::size_t& i, Common& c, bool* bad) {
  const std::string& a = args[i];
  const bool has_value = i + 1 < args.size();
  if (a == "-v") c.verbose = true;
  else if (a == "--threads" && has_value) c.mapper.threads = std::atoi(args[++i].c_str());
  else if (a == "--export" && has_value) c.export_dir = args[++i];
  else if (a == "--poses" && has_value) *bad = !pose_backend_from_name(args[++i], &c.mapper.pose_backend);
  else if (a == "--ba" && has_value) {
    const auto b = jba::backend_from_name(args[++i]);
    *bad = !b;
    if (b) c.mapper.ba_backend = *b;
  } else {
    return false;
  }
  return true;
}

demo::StepCallback printer(bool verbose) {
  return [verbose](const demo::Snapshot& s) {
    if (!verbose) return;
    std::printf("  %-10s +%zu images, %zu points, %.3f px\n", s.stage.c_str(), s.added.size(), s.points.size() / 3,
                s.mean_error_px);
  };
}

void print_times(const demo::Reconstruction& r) {
  std::printf("  poses on %s; first pair %d-%d (%u inliers, %.1f°); s: verify %.3f, tracks %.3f, pnp %.3f, "
              "triangulate %.3f, adjust %.3f, total %.3f\n",
              r.pose_backend.c_str(), r.first_a, r.first_b, r.first_inliers, r.first_angle_deg, r.time.verify,
              r.time.tracks, r.time.pnp, r.time.triangulate, r.time.adjust, r.time.total);
}

void export_model(const std::string& dir, const demo::Reconstruction& r, const std::vector<std::string>& names,
                  const std::vector<std::int32_t>& image_size) {
  if (!r.ok()) return;
  std::filesystem::create_directories(dir);
  jba::ColmapExportOptions eo;
  eo.image_size = image_size;
  for (std::int32_t i : r.image_of)
    eo.image_names.push_back(names.empty() ? "image_" + std::to_string(i) : names[static_cast<std::size_t>(i)]);
  jba::export_colmap(r.problem, dir, eo);
}

bool available(const demo::MapperOptions& mo) {
  if (jba::backend_available(mo.ba_backend)) return true;
  std::fprintf(stderr, "jba_demo: the %s backend is not available here\n", std::string(jba::backend_name(mo.ba_backend)).c_str());
  return false;
}

int synth(const std::vector<std::string>& args) {
  std::string scene_arg = "all", write_dir;
  demo::SceneOptions so;
  Common c;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string& a = args[i];
    const bool has_value = i + 1 < args.size();
    bool bad = false;
    if (common_option(args, i, c, &bad)) {
      if (bad) return usage();
    } else if (a == "--scene" && has_value) scene_arg = args[++i];
    else if (a == "--seed" && has_value) so.seed = std::strtoull(args[++i].c_str(), nullptr, 10);
    else if (a == "--noise" && has_value) so.noise_px = std::atof(args[++i].c_str());
    else if (a == "--wrong" && has_value) so.wrong_fraction = std::atof(args[++i].c_str());
    else if (a == "--false-pairs" && has_value) so.false_pairs = std::atoi(args[++i].c_str());
    else if (a == "--write" && has_value) write_dir = args[++i];
    else return usage();
  }
  demo::MapperOptions& mo = c.mapper;
  mo.seed = 7 + static_cast<std::uint32_t>(so.seed);
  demo::thresholds_for_noise(so.noise_px, mo);
  if (!available(mo)) return 2;
  const std::vector<std::string> names =
      scene_arg == "all" ? demo::scene_names() : std::vector<std::string>{scene_arg};

  int failed = 0;
  demo::Mapper mapper(mo);
  std::printf("%-6s %7s %6s %6s %6s %9s %9s %9s %9s %8s %8s %6s %7s  %s\n", "scene", "images", "pairs", "kept",
              "points", "reproj px", "rot med°", "ctr med%", "pts med%", "prec", "recall", "BA", "s", "result");
  for (const std::string& name : names) {
    const demo::Scene scene = demo::make_scene(name, so);
    if (c.verbose) std::printf("%s\n", name.c_str());
    const demo::Reconstruction r = mapper.run(scene.capture, printer(c.verbose));
    const demo::Metrics mt = demo::evaluate(scene, r);
    if (!mt.pass) ++failed;
    char images[32];
    std::snprintf(images, sizeof(images), "%zu/%zu", mt.registered, mt.images);
    std::printf("%-6s %7s %6zu %6d %6zu %9.3f %9.4f %9.4f %9.4f %7.1f%% %7.1f%% %6d %7.3f  %s%s%s\n", name.c_str(),
                images, scene.capture.pairs.size(), r.pairs_kept, mt.points, mt.mean_error_px, mt.rot_median_deg,
                mt.centre_median_pct, mt.point_median_pct, 100.0 * mt.mask_precision, 100.0 * mt.mask_recall,
                r.adjustments, r.time.total, mt.pass ? "PASS" : "FAIL", mt.why.empty() ? "" : ": ", mt.why.c_str());
    if (c.verbose) print_times(r);
    const demo::CaptureFile file = demo::to_capture_file(scene);
    if (!c.export_dir.empty())
      export_model((std::filesystem::path(c.export_dir) / name).string(), r, file.names, file.image_size);
    if (!write_dir.empty()) {
      std::filesystem::create_directories(write_dir);
      demo::write_capture((std::filesystem::path(write_dir) / (name + ".jbc")).string(), file);
    }
  }
  std::printf("  rot: rotation error after the best common rotation. ctr: camera-centre error, pts: point error,\n"
              "  both over the scene extent, after the similarity that best aligns the camera centres.\n"
              "  prec/recall: inlier masks of the kept pairs against the true matches.\n"
              "  Pass: all images, median rotation < 0.5°, median centre < 1 %%, mean reprojection < 1.5 px\n"
              "  or twice the keypoint noise.\n");
  return failed ? 1 : 0;
}

int run(const std::vector<std::string>& args) {
  if (args.empty()) return usage();
  const std::string path = args[0];
  std::string intrinsics = "reference", thresholds = "colmap";
  bool check = false;
  Common c;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string& a = args[i];
    const bool has_value = i + 1 < args.size();
    bool bad = false;
    if (common_option(args, i, c, &bad)) {
      if (bad) return usage();
    } else if (a == "--intrinsics" && has_value) intrinsics = args[++i];
    else if (a == "--thresholds" && has_value) thresholds = args[++i];
    else if (a == "--check") check = true;
    else return usage();
  }
  if ((intrinsics != "capture" && intrinsics != "reference") || (thresholds != "colmap" && thresholds != "demo"))
    return usage();
  demo::MapperOptions& mo = c.mapper;
  if (thresholds == "colmap") demo::colmap_thresholds(mo);
  if (!available(mo)) return 2;

  demo::CaptureFile file = demo::read_capture(path);
  const demo::Reference& ref = file.reference;
  // The reference's refined intrinsics, when it has them: both reconstructions then
  // share one calibration, as the mapper keeps intrinsics fixed.
  const bool ref_intrinsics = intrinsics == "reference" && !ref.intrinsics.empty();
  if (ref_intrinsics) file.capture.intrinsics = ref.intrinsics;
  std::size_t matches = 0;
  for (const demo::Pair& p : file.capture.pairs) matches += p.ka.size();
  std::printf("%s: %zu images, %zu intrinsics blocks, %zu pairs, %zu matches; intrinsics from the %s\n", path.c_str(),
              file.capture.num_images(),
              file.capture.intrinsics.size() / static_cast<std::size_t>(jba::num_params(file.capture.model)),
              file.capture.pairs.size(), matches, ref_intrinsics ? "reference" : "capture");

  demo::Mapper mapper(mo);
  const demo::Reconstruction r = mapper.run(file.capture, printer(c.verbose));
  if (!r.ok()) std::printf("  stopped: %s\n", r.error.c_str());
  std::printf("  pairs: %d kept of %zu (%d OK, %d without baseline, %d failed); %zu tracks; %d PnP calls, %d rejected\n",
              r.pairs_kept, file.capture.pairs.size(), r.pairs_ok, r.pairs_degenerate, r.pairs_failed, r.tracks,
              r.pnp_calls, r.pnp_rejected);
  print_times(r);
  if (!c.export_dir.empty()) export_model(c.export_dir, r, file.names, file.image_size);

  const std::size_t n_img = file.capture.num_images();
  if (ref.empty()) {
    std::size_t registered = 0;
    for (char g : r.registered) registered += g != 0;
    std::printf("  %zu/%zu images registered, %zu points, mean reprojection %.3f px (no reference to compare with)\n",
                registered, n_img, r.problem.num_points(), r.mean_error_px);
    return r.ok() ? 0 : 1;
  }
  const demo::Comparison cmp = demo::compare(file.capture, ref, r);
  std::printf("\n%-10s %10s %8s %8s %12s %10s\n", "", "registered", "points", "obs", "track length", "reproj px");
  const std::pair<const char*, demo::Side> sides[] = {{"ours", cmp.ours}, {"reference", cmp.reference}};
  for (const auto& [label, s] : sides)
    std::printf("%-10s %6zu/%-3zu %8zu %8zu %12.2f %10.3f\n", label, s.registered, n_img, s.points, s.observations,
                s.mean_track_length, s.mean_error_px);
  std::printf("\nOn the %zu images both registered: rotation error median %.4f°, max %.4f°; camera centre error\n"
              "median %.4f %% (%.4g units), max %.4f %% of the extent of the reference's centres.\n",
              cmp.common, cmp.rot_median_deg, cmp.rot_max_deg, cmp.centre_median_pct, cmp.centre_median,
              cmp.centre_max_pct);
  if (!check) return r.ok() ? 0 : 1;
  // --check: the thresholds of the generated scenes, for tests.
  const bool pass = r.ok() && cmp.common == cmp.reference.registered && cmp.rot_median_deg < 0.5 &&
                    cmp.centre_median_pct < 1.0;
  std::printf("check: %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  const std::string command = argv[1];
  const std::vector<std::string> args(argv + 2, argv + argc);
  try {
    if (command == "synth") return synth(args);
    if (command == "run") return run(args);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "jba_demo: %s\n", e.what());
    return 2;
  }
  return usage();
}
