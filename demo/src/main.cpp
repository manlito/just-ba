// jba_demo: the demo app's command line.
//
//   jba_demo synth [--scene orbit|wall|grid|all] [--seed n] [--noise px] [--wrong f] [--false-pairs n]
//                  [--focal-error f] [--priors sigma] [--write dir] [common options]
//     Reconstructs the generated scenes and compares each with its ground truth.
//     --focal-error: the mapper is given focal lengths that much too large (1.2: 2.2
//     times the truth, as COLMAP's guess on ETH3D), marked as a guess, so it searches
//     for them and refines them (--refine focal+distortion unless given). The truth
//     keeps the true ones; --write writes the given ones.
//     --priors: each image gets a position prior (D150), its true centre in another
//     frame (rotated, ten times larger, moved) with noise of sigma scene units per
//     axis; the result must then be in that frame: its centres and rotations are
//     compared with the truth carried into it, without fitting anything.
//     --write: also writes each scene as a capture file (<dir>/<scene>.jbc), truth as reference.
//
//   jba_demo run capture.jbc [--intrinsics capture|reference] [--thresholds colmap|demo] [--two-view-tracks]
//                [--tri-angle deg] [--json file] [--check] [--no-priors]
//                [--min-pair-inliers n] [--min-pair-coverage share]
//                [common options]
//     Reconstructs a capture file (from demo/tools/colmap_capture.py) and, when it holds a
//     reference reconstruction, compares with it (document 05, level 3).
//
//   common options: [--poses auto|cpu|metal|cuda|webgpu] [--ba cpu|metal|cuda|webgpu] [--threads n]
//                   [--refine none|focal|focal+distortion] [--export dir] [-v]
//     --refine: the intrinsics the adjustments refine. Default: focal+distortion with
//     --thresholds colmap, as COLMAP's mapper; none otherwise.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "align.h"
#include "capture_io.h"
#include "compare.h"
#include "evaluate.h"
#include "geometry.h"
#include "jba/export.h"
#include "mapper.h"
#include "scenes.h"

namespace {

int usage() {
  std::fprintf(stderr,
               "usage: jba_demo synth [--scene orbit|wall|grid|all] [--seed n] [--noise px] [--wrong f]\n"
               "                      [--false-pairs n] [--focal-error f] [--priors sigma] [--write dir] [options]\n"
               "       jba_demo run capture.jbc [--intrinsics capture|reference] [--thresholds colmap|demo]\n"
               "                    [--two-view-tracks] [--tri-angle deg] [--json file] [--check] [--no-priors]\n"
               "                    [--min-pair-inliers n] [--min-pair-coverage share]\n"
               "                    [options]\n"
               "options: [--poses auto|cpu|metal|cuda|webgpu] [--ba cpu|metal|cuda|webgpu] [--threads n]\n"
               "         [--refine none|focal|focal+distortion] [--export dir] [-v]\n");
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
  int refine = -1;  // jba::IntrinsicsGroup mask from --refine; -1: the command's default
};

bool refine_from_name(const std::string& s, int* mask) {
  if (s == "none") *mask = 0;
  else if (s == "focal") *mask = jba::kFocal;
  else if (s == "focal+distortion") *mask = jba::kFocal | jba::kDistortion;
  else return false;
  return true;
}

// Parses the options both commands take; false for one it does not know.
bool common_option(const std::vector<std::string>& args, std::size_t& i, Common& c, bool* bad) {
  const std::string& a = args[i];
  const bool has_value = i + 1 < args.size();
  if (a == "-v") c.verbose = true;
  else if (a == "--threads" && has_value) c.mapper.threads = std::atoi(args[++i].c_str());
  else if (a == "--export" && has_value) c.export_dir = args[++i];
  else if (a == "--poses" && has_value) *bad = !pose_backend_from_name(args[++i], &c.mapper.pose_backend);
  else if (a == "--refine" && has_value) *bad = !refine_from_name(args[++i], &c.refine);
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
  std::printf("  poses on %s; first pair %d-%d (%u inliers, %.1f°); s: focal search %.3f, verify %.3f, tracks %.3f, "
              "pnp %.3f, triangulate %.3f, adjust %.3f, total %.3f\n",
              r.pose_backend.c_str(), r.first_a, r.first_b, r.first_inliers, r.first_angle_deg, r.time.focal_search,
              r.time.verify,
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

// The comparison as JSON, for demo/tools/colmap_table.py.
void write_json(const std::string& out, const std::string& capture, const demo::Reconstruction& r,
                const demo::Comparison& c, std::size_t images, std::size_t truth_images, const demo::PoseAgreement& to,
                const demo::PoseAgreement& tr, double focal_given, double focal_reference) {
  std::FILE* f = std::fopen(out.c_str(), "w");
  if (!f) throw std::runtime_error("cannot write " + out);
  auto side = [&](const char* name, const demo::Side& s) {
    std::fprintf(f, "\"%s\": {\"registered\": %zu, \"points\": %zu, \"observations\": %zu, \"track_length\": %.6g, "
                 "\"reprojection_px\": %.6g}, ", name, s.registered, s.points, s.observations, s.mean_track_length, s.mean_error_px);
  };
  auto poses = [&](const char* name, const demo::PoseAgreement& a) {
    std::fprintf(f, "\"%s\": {\"common\": %zu, \"rot_median_deg\": %.6g, \"rot_max_deg\": %.6g, \"centre_median\": %.6g, "
                 "\"centre_median_pct\": %.6g, \"centre_max_pct\": %.6g}, ", name, a.common, a.rot_median_deg, a.rot_max_deg,
                 a.centre_median, a.centre_median_pct, a.centre_max_pct);
  };
  std::fprintf(f, "{\"capture\": \"%s\", \"images\": %zu, \"ok\": %s, ", capture.c_str(), images, r.ok() ? "true" : "false");
  side("ours", c.ours);
  side("reference", c.reference);
  demo::PoseAgreement vs;
  vs.common = c.common;
  vs.rot_median_deg = c.rot_median_deg;
  vs.rot_max_deg = c.rot_max_deg;
  vs.centre_median = c.centre_median;
  vs.centre_median_pct = c.centre_median_pct;
  vs.centre_max_pct = c.centre_max_pct;
  poses("ours_vs_reference", vs);
  if (truth_images) {
    std::fprintf(f, "\"truth_images\": %zu, ", truth_images);
    poses("ours_vs_truth", to);
    poses("reference_vs_truth", tr);
  }
  std::fprintf(f, "\"focal_given\": %.6g, \"focal_ours\": %.6g, \"focal_reference\": %.6g, ", focal_given,
               r.ok() ? demo::median_focal(r.problem.model, r.problem.intrinsics) : 0.0, focal_reference);
  std::fprintf(f, "\"reference_points_split\": %zu, \"our_points_joined\": %zu, \"seconds\": %.6g}\n",
               c.reference_points_split, c.our_points_joined, r.time.total);
  std::fclose(f);
}

// The prior frame of synth --priors: X_prior = s R X + t, the same for every scene.
struct PriorFrame {
  double R[9], t[3] = {120.0, -40.0, 15.0}, s = 10.0;
  PriorFrame() {
    demo::Rng rng(2026);
    demo::random_rotation(rng, 180.0, R);
  }
  void centre(const double* pose, double out[3]) const {
    double C[3];
    demo::camera_centre(pose, C);
    for (int r = 0; r < 3; ++r) out[r] = s * (R[3 * r] * C[0] + R[3 * r + 1] * C[1] + R[3 * r + 2] * C[2]) + t[r];
  }
};

// The reconstruction against the truth carried into the prior frame, nothing fitted:
// median centre error over the extent of the true centres there, median rotation error.
void prior_frame_errors(const demo::Scene& scene, const demo::Reconstruction& r, const PriorFrame& f,
                        double* centre_pct, double* rot_deg) {
  std::vector<double> truth, ce, re;
  for (std::size_t i = 0; i < scene.poses.size() / 7; ++i) {
    double C[3];
    f.centre(&scene.poses[7 * i], C);
    truth.insert(truth.end(), C, C + 3);
  }
  const double extent = demo::bbox_diagonal(truth);
  for (std::size_t i = 0; i < r.registered.size(); ++i) {
    if (!r.registered[i]) continue;
    double C[3], Rt[9], Re[9], Rg[9], St[9];
    demo::camera_centre(&r.poses[7 * i], C);
    ce.push_back(std::hypot(C[0] - truth[3 * i], C[1] - truth[3 * i + 1], C[2] - truth[3 * i + 2]) / extent * 100.0);
    demo::quat_to_rotmat(&r.poses[7 * i], Re);
    demo::quat_to_rotmat(&scene.poses[7 * i], Rt);
    demo::transpose3(f.R, St);
    demo::mat3_mul(Rt, St, Rg);  // world to camera, from the prior frame
    re.push_back(demo::rotation_angle_deg(Re, Rg));
  }
  *centre_pct = demo::median(ce);
  *rot_deg = demo::median(re);
}

int synth(const std::vector<std::string>& args) {
  std::string scene_arg = "all", write_dir;
  double focal_error = 0.0, prior_sigma = 0.0;
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
    else if (a == "--focal-error" && has_value) focal_error = std::atof(args[++i].c_str());
    else if (a == "--priors" && has_value) prior_sigma = std::atof(args[++i].c_str());
    else if (a == "--write" && has_value) write_dir = args[++i];
    else return usage();
  }
  demo::MapperOptions& mo = c.mapper;
  mo.seed = 7 + static_cast<std::uint32_t>(so.seed);
  demo::thresholds_for_noise(so.noise_px, mo);
  if (c.refine < 0 && focal_error != 0.0) c.refine = jba::kFocal | jba::kDistortion;
  mo.search_focal = focal_error != 0.0;
  if (c.refine >= 0) mo.refine_intrinsics = static_cast<std::uint8_t>(c.refine);
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
    demo::Capture start = scene.capture;  // what the mapper is given: focal lengths off by focal_error
    const int n_focal = start.model == jba::CameraModel::Pinhole ? 2 : 1;
    const std::size_t np = static_cast<std::size_t>(jba::num_params(start.model));
    for (std::size_t b = 0; b < start.intrinsics.size() / np; ++b)
      for (int k = 0; k < n_focal; ++k) start.intrinsics[np * b + k] *= 1.0 + focal_error;
    if (focal_error != 0.0) start.focal_unknown.assign(start.intrinsics.size() / np, 1);
    const PriorFrame frame;
    if (prior_sigma > 0.0) {
      demo::Rng rng(so.seed + 99);
      for (std::size_t i = 0; i < start.num_images(); ++i) {
        double C[3];
        frame.centre(&scene.poses[7 * i], C);
        for (int d = 0; d < 3; ++d) start.prior_position.push_back(C[d] + frame.s * prior_sigma * rng.normal());
        start.prior_sigma.insert(start.prior_sigma.end(), {frame.s * prior_sigma, frame.s * prior_sigma});
      }
    }
    const demo::Reconstruction r = mapper.run(start, printer(c.verbose));
    demo::Metrics mt = demo::evaluate(scene, r);
    double prior_centre_pct = 0.0, prior_rot_deg = 0.0;
    if (prior_sigma > 0.0 && r.ok()) {
      // In the prior frame, within the generated scenes' margins (1 % of the extent,
      // 0.5 degrees) and no fitted similarity to hide a wrong frame.
      prior_frame_errors(scene, r, frame, &prior_centre_pct, &prior_rot_deg);
      if (mt.pass && !r.aligned) {
        mt.pass = false;
        mt.why = "not in the priors' frame";
      } else if (mt.pass && (prior_centre_pct >= 1.0 || prior_rot_deg >= 0.5)) {
        mt.pass = false;
        mt.why = "off in the priors' frame";
      }
    }
    if (!mt.pass) ++failed;
    char images[32];
    std::snprintf(images, sizeof(images), "%zu/%zu", mt.registered, mt.images);
    std::printf("%-6s %7s %6zu %6d %6zu %9.3f %9.4f %9.4f %9.4f %7.1f%% %7.1f%% %6d %7.3f  %s%s%s\n", name.c_str(),
                images, scene.capture.pairs.size(), r.pairs_kept, mt.points, mt.mean_error_px, mt.rot_median_deg,
                mt.centre_median_pct, mt.point_median_pct, 100.0 * mt.mask_precision, 100.0 * mt.mask_recall,
                r.adjustments, r.time.total, mt.pass ? "PASS" : "FAIL", mt.why.empty() ? "" : ": ", mt.why.c_str());
    if (c.verbose) print_times(r);
    if (prior_sigma > 0.0)
      std::printf("       priors: %d of %zu images, %s; in their frame: centre median %.4f %%, rotation median %.4f°;"
                  " to the priors: %.4g horizontal, %.4g vertical (sigma %.4g)\n",
                  r.prior_images, mt.images, r.aligned ? "aligned" : "NOT aligned", prior_centre_pct, prior_rot_deg,
                  r.prior_error_horizontal, r.prior_error_vertical, frame.s * prior_sigma);
    if (focal_error != 0.0)
      std::printf("       focal: true %.1f px, given %.1f, searched x%.3f, refined %.1f\n",
                  demo::median_focal(scene.capture.model, scene.capture.intrinsics),
                  demo::median_focal(start.model, start.intrinsics), r.focal_factor,
                  r.ok() ? demo::median_focal(r.problem.model, r.problem.intrinsics) : 0.0);
    demo::CaptureFile file = demo::to_capture_file(scene);
    file.capture.intrinsics = start.intrinsics;
    file.capture.focal_unknown = start.focal_unknown;
    file.capture.prior_position = start.prior_position;
    file.capture.prior_sigma = start.prior_sigma;
    if (focal_error != 0.0) file.reference.intrinsics = scene.capture.intrinsics;  // the truth's
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
  bool check = false, two_view = false, no_priors = false;
  std::string json_path;
  double tri_angle = -1.0, min_pair_coverage = -1.0;
  int min_pair_inliers = -1;
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
    else if (a == "--json" && has_value) json_path = args[++i];
    else if (a == "--two-view-tracks") two_view = true;
    else if (a == "--no-priors") no_priors = true;
    else if (a == "--tri-angle" && has_value) tri_angle = std::atof(args[++i].c_str());
    else if (a == "--min-pair-inliers" && has_value) min_pair_inliers = std::atoi(args[++i].c_str());
    else if (a == "--min-pair-coverage" && has_value) min_pair_coverage = std::atof(args[++i].c_str());

    else return usage();
  }
  if ((intrinsics != "capture" && intrinsics != "reference") || (thresholds != "colmap" && thresholds != "demo"))
    return usage();
  demo::MapperOptions& mo = c.mapper;
  if (thresholds == "colmap") demo::colmap_thresholds(mo);
  if (c.refine >= 0) mo.refine_intrinsics = static_cast<std::uint8_t>(c.refine);
  if (two_view) mo.ignore_two_view_tracks = false;
  if (no_priors) mo.use_position_priors = false;
  if (tri_angle >= 0.0) mo.tri_max_angle_error_deg = tri_angle;
  if (min_pair_inliers >= 0) mo.min_pair_inliers = static_cast<std::uint32_t>(min_pair_inliers);
  if (min_pair_coverage >= 0.0) mo.min_pair_coverage = min_pair_coverage;
  if (!available(mo)) return 2;

  demo::CaptureFile file = demo::read_capture(path);
  const demo::Reference& ref = file.reference;
  // The reference's refined intrinsics, when it has them: both reconstructions then
  // start from one calibration (and keep it, with --refine none).
  const bool ref_intrinsics = intrinsics == "reference" && !ref.intrinsics.empty();
  if (ref_intrinsics) {
    file.capture.intrinsics = ref.intrinsics;
    file.capture.focal_unknown.clear();  // the reference's focal lengths are not a guess
  }
  std::size_t matches = 0;
  for (const demo::Pair& p : file.capture.pairs) matches += p.ka.size();
  const char* refined = mo.refine_intrinsics == 0                ? "kept fixed"
                        : mo.refine_intrinsics == jba::kFocal ? "focal refined"
                                                              : "focal and distortion refined";
  std::printf("%s: %zu images, %zu intrinsics blocks, %zu pairs, %zu matches; intrinsics from the %s, %s\n",
              path.c_str(), file.capture.num_images(),
              file.capture.intrinsics.size() / static_cast<std::size_t>(jba::num_params(file.capture.model)),
              file.capture.pairs.size(), matches, ref_intrinsics ? "reference" : "capture", refined);

  demo::Mapper mapper(mo);
  const demo::Reconstruction r = mapper.run(file.capture, printer(c.verbose));
  if (!r.ok()) std::printf("  stopped: %s\n", r.error.c_str());
  std::printf("  pairs: %d kept of %zu (%d OK, %d without baseline, %d failed, %d too narrow); %zu tracks; %d PnP"
              " calls, %d rejected; %d model%s grown\n",
              r.pairs_kept, file.capture.pairs.size(), r.pairs_ok, r.pairs_degenerate, r.pairs_failed, r.pairs_narrow,
              r.tracks, r.pnp_calls, r.pnp_rejected, r.models, r.models == 1 ? "" : "s");
  for (std::size_t k = 0; r.all_models.size() > 1 && k < r.all_models.size(); ++k) {
    const demo::Reconstruction::Model& md = r.all_models[k];
    std::size_t n = 0;
    for (char g : md.registered) n += g != 0;
    std::printf("    model %zu%s: %zu images, %zu points, %.3f px\n", k + 1,
                static_cast<int>(k) == r.kept_model ? " (kept)" : "", n, md.points.size() / 3, md.mean_error_px);
  }
  print_times(r);
  if (!file.capture.prior_position.empty()) {
    std::size_t with = 0;
    for (std::size_t i = 0; i < file.capture.num_images(); ++i) with += file.capture.has_prior(i);
    if (no_priors)
      std::printf("  position priors: %zu images have one, not used (--no-priors)\n", with);
    else
      std::printf("  position priors: %zu images have one, %d registered; %s; %d within %.3g sigma, median distance"
                  " %.4g horizontal, %.4g vertical\n",
                  with, r.prior_images, r.aligned ? "aligned to them" : "NOT aligned (too few, or along a line)",
                  r.prior_inliers, mo.prior_inlier_sigmas, r.prior_error_horizontal, r.prior_error_vertical);
  }
  std::printf("  tracks: %zu built, %zu of them with two keypoints in one image; %zu after splitting\n",
              r.track_stats.components, r.track_stats.conflicted, r.all_tracks.size());
  if (!c.export_dir.empty()) export_model(c.export_dir, r, file.names, file.image_size);
  if (r.ok() && mo.refine_intrinsics) {
    const double f0 = demo::median_focal(file.capture.model, file.capture.intrinsics);
    const double f1 = demo::median_focal(r.problem.model, r.problem.intrinsics);
    std::printf("  focal (median over blocks): given %.1f px", f0);
    if (r.focal_factor != 1.0) std::printf(", searched x%.3f", r.focal_factor);
    std::printf(", refined %.1f", f1);
    if (!ref.intrinsics.empty())
      std::printf(", the reference's %.1f (ours %+.2f %%)", demo::median_focal(file.capture.model, ref.intrinsics),
                  100.0 * (f1 / demo::median_focal(file.capture.model, ref.intrinsics) - 1.0));
    std::printf("\n");
  }

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
  std::printf("\nTracks: %zu observations on both sides, %zu ours only, %zu the reference's only.\n"
              "%zu reference points have shared observations; %zu of them (%.1f %%) are split over several of\n"
              "ours, %.3f of ours each on average. %zu of our points share observations; %zu (%.1f %%) join\n"
              "several reference points.\n",
              cmp.shared_obs, cmp.ours_only_obs, cmp.reference_only_obs, cmp.reference_points_seen,
              cmp.reference_points_split,
              cmp.reference_points_seen ? 100.0 * cmp.reference_points_split / cmp.reference_points_seen : 0.0,
              cmp.ours_per_reference_point, cmp.our_points_seen, cmp.our_points_joined,
              cmp.our_points_seen ? 100.0 * cmp.our_points_joined / cmp.our_points_seen : 0.0);
  std::printf("The reference's observations we lack: %zu filtered from one of our points, %zu in a track of ours\n"
              "with no point, %zu in no track of ours. Our points sharing nothing with the reference: %zu\n"
              "(%zu of them seen twice).\n",
              cmp.missing_filtered, cmp.missing_untriangulated, cmp.missing_untracked, cmp.ours_only_points,
              cmp.ours_only_two_view);
  demo::PoseAgreement truth_ours, truth_ref;
  if (!file.truth_registered.empty()) {
    const demo::PoseAgreement o = demo::compare_poses(r.registered, r.poses, file.truth_registered, file.truth_poses);
    const demo::PoseAgreement c2 = demo::compare_poses(ref.registered, ref.poses, file.truth_registered, file.truth_poses);
    truth_ours = o;
    truth_ref = c2;
    std::size_t in_truth = 0;
    for (char g : file.truth_registered) in_truth += g != 0;
    std::printf("\nAgainst the ground truth (%zu images), after the same alignment; centre error in the truth's units:\n",
                in_truth);
    const std::pair<const char*, demo::PoseAgreement> rows[] = {{"ours", o}, {"reference", c2}};
    for (const auto& [label, a] : rows)
      std::printf("%-10s %3zu images: rotation median %.4f°, max %.4f°; centre median %.4g (%.4f %%), max %.4f %%\n",
                  label, a.common, a.rot_median_deg, a.rot_max_deg, a.centre_median, a.centre_median_pct, a.centre_max_pct);
  }
  std::size_t truth_images = 0;
  for (char g : file.truth_registered) truth_images += g != 0;
  if (!json_path.empty()) write_json(json_path, path, r, cmp, n_img, truth_images, truth_ours, truth_ref,
                                     demo::median_focal(file.capture.model, file.capture.intrinsics),
                                     ref.intrinsics.empty() ? 0.0 : demo::median_focal(file.capture.model, ref.intrinsics));
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
