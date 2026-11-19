// The live demo's WebAssembly entry points (D156). The page finds keypoints (ALIKED,
// just-keypoints) and matches every pair (LightGlue, just-matching) on its images, then
// hands them here: the mapper reconstructs them with poselib-gpu and this library, on
// the CPU or WebGPU, and reports as it goes. demo_reconstruct and demo_backends suspend
// on WebGPU (JSPI), so JavaScript calls them with ccall's {async: true}.
//
// The answer is JSON (demo_reconstruct's return value) plus binary arrays read from the
// module's memory: demo_points() (float, 3 per point, every step's points one after the
// other), demo_tracks() (int32, each of those points' track) and demo_colour_obs()
// (int32, per point of the last step: an image and a keypoint that see it).
#include <emscripten/emscripten.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "geometry.h"
#include "mapper.h"

namespace {

std::string result;  // the last answer; valid until the next call
// Kept, one per pose backend: on WebGPU the device start-up is paid once.
std::map<std::int32_t, std::unique_ptr<demo::Mapper>> mappers;
std::vector<float> step_points;
std::vector<std::int32_t> step_tracks;
std::vector<std::int32_t> colour_obs;

std::string escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    if (c == '\n') {
      o += "\\n";
      continue;
    }
    o += c;
  }
  return o;
}

void num(std::ostringstream& j, double v, const char* fmt = "%.6g") {
  if (!std::isfinite(v)) {
    j << "null";
    return;
  }
  char buf[32];
  std::snprintf(buf, sizeof(buf), fmt, v);
  j << buf;
}

// Camera centre and world-to-camera rotation (row-major) of a pose qw qx qy qz tx ty tz.
void write_camera(std::ostringstream& j, const double* pose) {
  double C[3], R[9];
  demo::camera_centre(pose, C);
  demo::quat_to_rotmat(pose, R);
  j << "[";
  for (int k = 0; k < 3; ++k) {
    if (k) j << ",";
    num(j, C[k], "%.6g");
  }
  for (int k = 0; k < 9; ++k) {
    j << ",";
    num(j, R[k], "%.5g");
  }
  j << "]";
}

struct Step {
  std::string stage;
  std::vector<std::int32_t> added;
  std::vector<double> poses;
  std::vector<char> registered;
  std::size_t first_point = 0, num_points = 0;
  double mean_error_px = 0, seconds = 0;
};

// Hands the page where the run is, lets it draw, and asks whether to go on (0: stop).
// The wait is one task, posted through a MessageChannel: unlike setTimeout's, it is not
// throttled to once a second when the page is in the background, so a hidden tab costs
// the run nothing.
EM_ASYNC_JS(int, demo_report, (const char* json), {
  const go = Module.onProgress ? Module.onProgress(UTF8ToString(json)) !== false : true;
  await new Promise(resolve => {
    const ch = new MessageChannel();
    ch.port1.onmessage = () => resolve();
    ch.port2.postMessage(0);
  });
  return go ? 1 : 0;
});

}  // namespace

extern "C" {

// {"poses": [backends poselib-gpu can use], "ba": [backends the solver can use]}
EMSCRIPTEN_KEEPALIVE const char* demo_backends() {
  std::ostringstream j;
  j << "{\"poses\": [\"cpu\"" << (jpo_backend_available(JPO_BACKEND_WEBGPU) ? ", \"webgpu\"" : "")
    << "], \"ba\": [\"cpu\"" << (jba::backend_available(jba::BackendKind::WebGpu) ? ", \"webgpu\"" : "") << "]}";
  result = j.str();
  return result.c_str();
}

EMSCRIPTEN_KEEPALIVE const float* demo_points() { return step_points.data(); }
EMSCRIPTEN_KEEPALIVE const std::int32_t* demo_tracks() { return step_tracks.data(); }
EMSCRIPTEN_KEEPALIVE const std::int32_t* demo_colour_obs() { return colour_obs.data(); }

// Reconstructs n images from the page's keypoints and matches, with COLMAP's mapper
// thresholds and its refinement of focal length and distortion (SIMPLE_RADIAL).
//   size: 2 per image, width and height in pixels
//   focal: per image, its focal length in pixels if known; NaN or <= 0: unknown, then
//          1.2 times the larger side, searched for and refined (D144)
//   kp_begin: n + 1 offsets into kp_xy (2 floats per keypoint, COLMAP's convention: the
//          centre of the first pixel at 0.5, 0.5)
//   pair_images: 2 per pair; pair_begin: n_pairs + 1 offsets into matches (2 per match:
//          a keypoint of the first image, one of the second)
//   poses: "cpu" or "webgpu" (poselib-gpu); ba: "cpu" or "webgpu" (the solver)
EMSCRIPTEN_KEEPALIVE const char* demo_reconstruct(int n, const std::int32_t* size, const double* focal,
                                                  const std::int32_t* kp_begin, const float* kp_xy, int n_pairs,
                                                  const std::int32_t* pair_images, const std::int32_t* pair_begin,
                                                  const std::int32_t* matches, const char* poses, const char* ba) {
  step_points.clear();
  step_tracks.clear();
  colour_obs.clear();
  try {
    demo::MapperOptions mo;
    demo::colmap_thresholds(mo);
    const std::string pb = poses;
    mo.pose_backend = pb == "webgpu" ? JPO_BACKEND_WEBGPU : JPO_BACKEND_CPU;
    const auto bb = jba::backend_from_name(ba);
    if (!bb) throw std::invalid_argument(std::string("unknown backend ") + ba);
    mo.ba_backend = *bb;
    mo.threads = 1;  // no threads in this build
    auto& mapper = mappers[mo.pose_backend];
    if (!mapper) mapper = std::make_unique<demo::Mapper>(mo);
    mapper->set_options(mo);

    // One SIMPLE_RADIAL block per image size and focal length.
    demo::Capture c;
    c.model = jba::CameraModel::SimpleRadial;
    std::map<std::tuple<int, int, double>, std::int32_t> blocks;
    for (int i = 0; i < n; ++i) {
      const int w = size[2 * i], h = size[2 * i + 1];
      const bool known = std::isfinite(focal[i]) && focal[i] > 0;
      const double f = known ? focal[i] : 1.2 * std::max(w, h);
      const auto key = std::make_tuple(w, h, f);
      auto it = blocks.find(key);
      if (it == blocks.end()) {
        it = blocks.emplace(key, static_cast<std::int32_t>(blocks.size())).first;
        c.intrinsics.insert(c.intrinsics.end(), {f, 0.5 * w, 0.5 * h, 0.0});
        c.focal_unknown.push_back(known ? 0 : 1);
        c.image_size.insert(c.image_size.end(), {w, h});
      }
      c.image_intrinsics.push_back(it->second);
      c.keypoints.emplace_back(kp_xy + 2 * kp_begin[i], kp_xy + 2 * kp_begin[i + 1]);
    }
    std::size_t total_matches = 0;
    for (int p = 0; p < n_pairs; ++p) {
      demo::Pair pr;
      pr.a = pair_images[2 * p];
      pr.b = pair_images[2 * p + 1];
      for (std::int32_t m = pair_begin[p]; m < pair_begin[p + 1]; ++m) {
        pr.ka.push_back(matches[2 * m]);
        pr.kb.push_back(matches[2 * m + 1]);
      }
      total_matches += pr.ka.size();
      c.pairs.push_back(std::move(pr));
    }

    // Live progress, at most four times a second and at every new stage.
    const auto t0 = std::chrono::steady_clock::now();
    auto last = t0 - std::chrono::seconds(1);
    std::string last_stage;
    const auto on_progress = [&](const demo::Progress& p) {
      const auto now = std::chrono::steady_clock::now();
      if (p.stage == last_stage && now - last < std::chrono::milliseconds(250)) return true;
      last = now;
      last_stage = p.stage;
      std::ostringstream j;
      j << "{\"stage\": \"" << p.stage << "\", \"model\": " << p.model << ", \"registered\": " << p.registered
        << ", \"images\": " << p.images << ", \"adjustments\": " << p.adjustments << ", \"iteration\": "
        << p.iteration << ", \"seconds\": ";
      num(j, std::chrono::duration<double>(now - t0).count());
      j << "}";
      return demo_report(j.str().c_str()) != 0;
    };

    std::vector<Step> steps;
    const demo::Reconstruction r = mapper->run(
        c,
        [&](const demo::Snapshot& s) {
          Step st;
          st.stage = s.stage;
          st.added = s.added;
          st.poses = *s.poses;
          st.registered = *s.registered;
          st.first_point = step_points.size() / 3;
          st.num_points = s.points.size() / 3;
          for (double v : s.points) step_points.push_back(static_cast<float>(v));
          step_tracks.insert(step_tracks.end(), s.point_track.begin(), s.point_track.end());
          st.mean_error_px = s.mean_error_px;
          st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
          steps.push_back(std::move(st));
        },
        on_progress);

    // Per point of the result, one observation to colour it by.
    const std::size_t np = r.problem.points.size() / 3;
    colour_obs.assign(2 * np, -1);
    for (std::size_t o = 0; o < r.problem.obs_point.size(); ++o) {
      const auto p = static_cast<std::size_t>(r.problem.obs_point[o]);
      if (colour_obs[2 * p] >= 0) continue;
      colour_obs[2 * p] = r.image_of[static_cast<std::size_t>(r.problem.obs_image[o])];
      colour_obs[2 * p + 1] = r.obs_keypoint[o];
    }

    std::ostringstream j;
    j << "{\"ok\": " << (r.ok() ? "true" : "false") << ", \"cancelled\": " << (r.cancelled ? "true" : "false")
      << ", \"error\": \"" << escape(r.error) << "\", \"images\": " << n << ", \"pairs\": " << n_pairs
      << ", \"matches\": " << total_matches << ", \"poses_backend\": \"" << escape(r.pose_backend)
      << "\", \"ba_backend\": \"" << jba::backend_name(mo.ba_backend) << "\"";
    int registered = 0;
    for (char g : r.registered) registered += g ? 1 : 0;
    j << ", \"registered\": " << registered << ", \"points\": " << np
      << ", \"observations\": " << r.problem.obs_point.size() << ", \"mean_error_px\": ";
    num(j, r.mean_error_px);
    j << ", \"models\": " << r.models << ", \"focal_factor\": ";
    num(j, r.focal_factor);
    // The refined intrinsics: f, cx, cy, k per block.
    j << ", \"intrinsics\": [";
    for (std::size_t k = 0; k < r.problem.intrinsics.size(); ++k) {
      if (k) j << ",";
      num(j, r.problem.intrinsics[k]);
    }
    j << "], \"image_block\": [";
    for (int i = 0; i < n; ++i) j << (i ? "," : "") << c.block(static_cast<std::size_t>(i));
    j << "], \"pair_inliers\": [";
    for (std::size_t p = 0; p < r.pair_results.size(); ++p) j << (p ? "," : "") << r.pair_results[p].n_inliers;
    j << "], \"pair_kept\": [";
    for (std::size_t p = 0; p < r.pair_kept.size(); ++p) j << (p ? "," : "") << (r.pair_kept[p] ? 1 : 0);
    j << "], \"image_error_px\": [";
    for (std::size_t i = 0; i < r.image_error_px.size(); ++i) {
      if (i) j << ",";
      num(j, r.image_error_px[i], "%.4g");
    }
    j << "], \"image_points\": [";
    for (std::size_t i = 0; i < r.image_points_seen.size(); ++i) j << (i ? "," : "") << r.image_points_seen[i];
    j << "], \"steps\": [";
    for (std::size_t k = 0; k < steps.size(); ++k) {
      const Step& st = steps[k];
      j << (k ? "," : "") << "{\"stage\": \"" << st.stage << "\", \"seconds\": ";
      num(j, st.seconds);
      j << ", \"mean_error_px\": ";
      num(j, st.mean_error_px);
      j << ", \"first_point\": " << st.first_point << ", \"num_points\": " << st.num_points << ", \"added\": [";
      for (std::size_t a = 0; a < st.added.size(); ++a) j << (a ? "," : "") << st.added[a];
      j << "], \"cameras\": [";
      bool first = true;
      for (std::size_t i = 0; i < st.registered.size(); ++i) {
        if (!st.registered[i]) continue;
        j << (first ? "" : ",") << "{\"image\": " << i << ", \"pose\": ";
        write_camera(j, &st.poses[7 * i]);
        j << "}";
        first = false;
      }
      j << "]}";
    }
    j << "], \"pairs_ok\": " << r.pairs_ok << ", \"pairs_degenerate\": " << r.pairs_degenerate
      << ", \"pairs_failed\": " << r.pairs_failed << ", \"pairs_kept\": " << r.pairs_kept << ", \"tracks\": " << r.tracks
      << ", \"first_pair\": [" << r.first_a << "," << r.first_b << "], \"pnp_calls\": " << r.pnp_calls
      << ", \"pnp_rejected\": " << r.pnp_rejected << ", \"adjustments\": " << r.adjustments;
    j << ", \"time\": {\"focal_search\": ";
    num(j, r.time.focal_search);
    j << ", \"verify\": ";
    num(j, r.time.verify);
    j << ", \"tracks\": ";
    num(j, r.time.tracks);
    j << ", \"pnp\": ";
    num(j, r.time.pnp);
    j << ", \"triangulate\": ";
    num(j, r.time.triangulate);
    j << ", \"adjust\": ";
    num(j, r.time.adjust);
    j << ", \"total\": ";
    num(j, r.time.total);
    j << "}}";
    result = j.str();
  } catch (const std::exception& e) {
    result = std::string("{\"ok\": false, \"error\": \"") + escape(e.what()) + "\"}";
  }
  return result.c_str();
}

}  // extern "C"
