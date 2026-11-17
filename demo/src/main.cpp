// jba_demo: the demo app's command line. Reconstructs the generated scenes with the
// incremental mapper and compares each with its ground truth.
//
//   jba_demo synth [--scene orbit|wall|grid|all] [--seed n] [--poses auto|cpu|metal|cuda|webgpu]
//                  [--ba cpu|metal|cuda|webgpu] [--threads n] [--export dir] [-v]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

#include "evaluate.h"
#include "jba/export.h"
#include "mapper.h"
#include "scenes.h"

namespace {

int usage() {
  std::fprintf(stderr,
               "usage: jba_demo synth [--scene orbit|wall|grid|all] [--seed n] [--poses auto|cpu|metal|cuda|webgpu]\n"
               "                      [--ba cpu|metal|cuda|webgpu] [--threads n] [--export dir] [-v]\n");
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

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || std::strcmp(argv[1], "synth") != 0) return usage();
  std::string scene_arg = "all", export_dir;
  demo::SceneOptions so;
  demo::MapperOptions mo;
  bool verbose = false;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    const bool has_value = i + 1 < argc;
    if (a == "-v") verbose = true;
    else if (a == "--scene" && has_value) scene_arg = argv[++i];
    else if (a == "--seed" && has_value) so.seed = std::strtoull(argv[++i], nullptr, 10);
    else if (a == "--threads" && has_value) mo.threads = std::atoi(argv[++i]);
    else if (a == "--noise" && has_value) so.noise_px = std::atof(argv[++i]);
    else if (a == "--wrong" && has_value) so.wrong_fraction = std::atof(argv[++i]);
    else if (a == "--false-pairs" && has_value) so.false_pairs = std::atoi(argv[++i]);
    else if (a == "--export" && has_value) export_dir = argv[++i];
    else if (a == "--poses" && has_value) {
      if (!pose_backend_from_name(argv[++i], &mo.pose_backend)) return usage();
    } else if (a == "--ba" && has_value) {
      const auto b = jba::backend_from_name(argv[++i]);
      if (!b) return usage();
      mo.ba_backend = *b;
    } else {
      return usage();
    }
  }
  mo.seed = 7 + static_cast<std::uint32_t>(so.seed);
  demo::thresholds_for_noise(so.noise_px, mo);
  if (!jba::backend_available(mo.ba_backend)) {
    std::fprintf(stderr, "jba_demo: the %s backend is not available here\n",
                 std::string(jba::backend_name(mo.ba_backend)).c_str());
    return 2;
  }
  std::vector<std::string> names = scene_arg == "all" ? demo::scene_names() : std::vector<std::string>{scene_arg};

  int failed = 0;
  try {
    demo::Mapper mapper(mo);
    std::printf("%-6s %7s %6s %6s %6s %9s %9s %9s %9s %8s %8s %6s %7s  %s\n", "scene", "images", "pairs", "kept",
                "points", "reproj px", "rot med°", "ctr med%", "pts med%", "prec", "recall", "BA", "s", "result");
    for (const std::string& name : names) {
      const demo::Scene scene = demo::make_scene(name, so);
      const demo::StepCallback step = [&](const demo::Snapshot& s) {
        if (!verbose) return;
        std::printf("  %-10s +%zu images, %zu points, %.3f px\n", s.stage.c_str(), s.added.size(), s.points.size() / 3,
                    s.mean_error_px);
      };
      if (verbose) std::printf("%s\n", name.c_str());
      const demo::Reconstruction r = mapper.run(scene.capture, step);
      const demo::Metrics mt = demo::evaluate(scene, r);
      if (!mt.pass) ++failed;
      char images[32];
      std::snprintf(images, sizeof(images), "%zu/%zu", mt.registered, mt.images);
      std::printf("%-6s %7s %6zu %6d %6zu %9.3f %9.4f %9.4f %9.4f %7.1f%% %7.1f%% %6d %7.3f  %s%s%s\n", name.c_str(),
                  images, scene.capture.pairs.size(), r.pairs_kept, mt.points, mt.mean_error_px, mt.rot_median_deg,
                  mt.centre_median_pct, mt.point_median_pct, 100.0 * mt.mask_precision, 100.0 * mt.mask_recall,
                  r.adjustments, r.time.total, mt.pass ? "PASS" : "FAIL", mt.why.empty() ? "" : ": ", mt.why.c_str());
      if (verbose)
        std::printf("  poses on %s; first pair %d-%d (%u inliers, %.1f°); s: verify %.3f, tracks %.3f, pnp %.3f, "
                    "triangulate %.3f, adjust %.3f\n",
                    r.pose_backend.c_str(), r.first_a, r.first_b, r.first_inliers, r.first_angle_deg, r.time.verify,
                    r.time.tracks, r.time.pnp, r.time.triangulate, r.time.adjust);
      if (!export_dir.empty() && r.ok()) {
        const std::filesystem::path dir = std::filesystem::path(export_dir) / name;
        std::filesystem::create_directories(dir);
        jba::ColmapExportOptions eo;
        eo.image_size = {static_cast<std::int32_t>(scene.width), static_cast<std::int32_t>(scene.height)};
        for (std::int32_t i : r.image_of) eo.image_names.push_back("image_" + std::to_string(i) + ".jpg");
        jba::export_colmap(r.problem, dir.string(), eo);
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "jba_demo: %s\n", e.what());
    return 2;
  }
  std::printf("  rot: rotation error after the best common rotation. ctr: camera-centre error, pts: point error,\n"
              "  both over the scene extent, after the similarity that best aligns the camera centres.\n"
              "  prec/recall: inlier masks of the kept pairs against the true matches.\n"
              "  Pass: all images, median rotation < 0.5°, median centre < 1 %%, mean reprojection < 1.5 px\n"
              "  or twice the keypoint noise.\n");
  return failed ? 1 : 0;
}
