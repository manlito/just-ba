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
//
// The live capture (D157) grows a map frame by frame with demo::Session (D153), in a
// worker: demo_session_new, demo_session_add, demo_session_retry, demo_session_snapshot
// and demo_session_finish, the same binary arrays after a snapshot or the finish. On
// the page's own thread, demo_pair_inliers checks a live frame's matches against a
// relative pose on the CPU, as Like Recon's capture loop does.
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
#include "jpo/jpo.h"
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

// Per point of `r`, one observation (image, keypoint) to colour it by.
void colour_of(const demo::Reconstruction& r) {
  const std::size_t np = r.problem.points.size() / 3;
  colour_obs.assign(2 * np, -1);
  for (std::size_t o = 0; o < r.problem.obs_point.size(); ++o) {
    const auto p = static_cast<std::size_t>(r.problem.obs_point[o]);
    if (colour_obs[2 * p] >= 0) continue;
    colour_obs[2 * p] = r.image_of[static_cast<std::size_t>(r.problem.obs_image[o])];
    colour_obs[2 * p + 1] = r.obs_keypoint[o];
  }
}

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

    colour_of(r);
    const std::size_t np = r.problem.points.size() / 3;

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

namespace {

// ---- the live capture's session (D157) ------------------------------------------------

std::unique_ptr<demo::Mapper> session_mapper;  // the session's own: it must outlive it
std::unique_ptr<demo::Session> session;
jpo_ctx* live_ctx = nullptr;  // demo_pair_inliers' context, CPU

// Pairs from the page: per pair the earlier image, n_pairs + 1 offsets into matches
// (2 per match: a keypoint of the earlier image, one of `image`).
std::vector<demo::Pair> pairs_from(int n_pairs, const std::int32_t* earlier, const std::int32_t* begin,
                                   const std::int32_t* matches, std::int32_t image) {
  std::vector<demo::Pair> out;
  for (int p = 0; p < n_pairs; ++p) {
    demo::Pair pr;
    pr.a = earlier[p];
    pr.b = image;
    for (std::int32_t m = begin[p]; m < begin[p + 1]; ++m) {
      pr.ka.push_back(matches[2 * m]);
      pr.kb.push_back(matches[2 * m + 1]);
    }
    out.push_back(std::move(pr));
  }
  return out;
}

void write_ints(std::ostringstream& j, const std::vector<std::int32_t>& v) {
  j << "[";
  for (std::size_t k = 0; k < v.size(); ++k) j << (k ? "," : "") << v[k];
  j << "]";
}

std::string step_json(const demo::SessionStep& st) {
  std::ostringstream j;
  j << "{\"ok\": " << (st.ok() ? "true" : "false") << ", \"error\": \"" << escape(st.error) << "\", \"image\": " << st.image
    << ", \"started\": " << (st.started ? "true" : "false") << ", \"piece\": " << st.piece
    << ", \"rebuilt\": " << (st.rebuilt ? "true" : "false") << ", \"rejoined\": " << (st.rejoined ? "true" : "false")
    << ", \"placed\": ";
  write_ints(j, st.placed);
  j << ", \"window\": ";
  write_ints(j, st.window);
  j << ", \"pairs_kept\": " << st.pairs_kept << ", \"pnp_inliers\": " << st.pnp_inliers
    << ", \"points_seen\": " << st.points_seen << ", \"mean_error_px\": ";
  num(j, st.mean_error_px);
  j << ", \"seconds\": ";
  num(j, st.seconds);
  j << ", \"pieces\": " << session->num_pieces() << ", \"largest\": " << session->largest_piece() << "}";
  return j.str();
}

// The registered images of a model and their poses, JSON.
void write_cameras(std::ostringstream& j, const std::vector<char>& registered, const std::vector<double>& poses) {
  j << "[";
  bool first = true;
  for (std::size_t i = 0; i < registered.size() && 7 * i + 7 <= poses.size(); ++i) {
    if (!registered[i]) continue;
    j << (first ? "" : ",") << "{\"image\": " << i << ", \"pose\": ";
    write_camera(j, &poses[7 * i]);
    j << "}";
    first = false;
  }
  j << "]";
}

}  // namespace

extern "C" {

// A session for a capture of width x height frames, one SIMPLE_RADIAL camera starting
// at focal f (a guess when focal_unknown: searched for once the map starts, D144), with
// COLMAP's thresholds as Like Recon's online map has them: min_pair_inliers for its
// pairs, `window` frames adjusted together, a new piece after `restart_after` frames in
// a row not placed, the map starting from the widest pair it can (min angle 0).
EMSCRIPTEN_KEEPALIVE const char* demo_session_new(int width, int height, double f, int focal_unknown, int window,
                                                 int restart_after, int min_pair_inliers, const char* poses,
                                                 const char* ba) {
  try {
    session.reset();
    demo::MapperOptions mo;
    demo::colmap_thresholds(mo);
    const std::string pb = poses;
    mo.pose_backend = pb == "webgpu" ? JPO_BACKEND_WEBGPU : JPO_BACKEND_CPU;
    const auto bb = jba::backend_from_name(ba);
    if (!bb) throw std::invalid_argument(std::string("unknown backend ") + ba);
    mo.ba_backend = *bb;
    mo.threads = 1;
    if (min_pair_inliers > 0) mo.min_pair_inliers = static_cast<std::uint32_t>(min_pair_inliers);
    if (!session_mapper || session_mapper->options().pose_backend != mo.pose_backend)
      session_mapper = std::make_unique<demo::Mapper>(mo);
    session_mapper->set_options(mo);
    demo::SessionOptions so;
    so.window = std::max(0, window);
    if (restart_after > 0) so.restart_after = restart_after;
    so.min_start_angle_deg = 0.0;
    session = std::make_unique<demo::Session>(*session_mapper, jba::CameraModel::SimpleRadial,
                                              std::vector<double>{f, 0.5 * width, 0.5 * height, 0.0},
                                              focal_unknown != 0, width, height, so);
    result = "{\"ok\": true}";
  } catch (const std::exception& e) {
    result = std::string("{\"ok\": false, \"error\": \"") + escape(e.what()) + "\"}";
  }
  return result.c_str();
}

// The next kept frame: n_kp keypoints xy (2 floats each, just-keypoints' convention, the
// centre of the first pixel at 0, 0) and its matches to earlier frames of the session.
// defer_split: not placed, the page will retry with more pairs before a new piece starts.
EMSCRIPTEN_KEEPALIVE const char* demo_session_add(int n_kp, const float* xy, int n_pairs, const std::int32_t* earlier,
                                                 const std::int32_t* pair_begin, const std::int32_t* matches,
                                                 int defer_split) {
  try {
    if (!session) throw std::runtime_error("no session");
    demo::SessionImage im;
    im.keypoints.reserve(2 * static_cast<std::size_t>(n_kp));
    for (int k = 0; k < 2 * n_kp; ++k) im.keypoints.push_back(static_cast<double>(xy[k]) + 0.5);  // the capture's convention
    im.pairs = pairs_from(n_pairs, earlier, pair_begin, matches, static_cast<std::int32_t>(session->num_images()));
    im.defer_split = defer_split != 0;
    result = step_json(session->add(im));
  } catch (const std::exception& e) {
    result = std::string("{\"ok\": false, \"error\": \"") + escape(e.what()) + "\"}";
  }
  return result.c_str();
}

// The last frame again with more pairs (chosen by appearance), or a new piece from it at
// once with new_piece (a Shoot now that can't be placed).
EMSCRIPTEN_KEEPALIVE const char* demo_session_retry(int n_pairs, const std::int32_t* earlier,
                                                   const std::int32_t* pair_begin, const std::int32_t* matches,
                                                   int new_piece) {
  try {
    if (!session || !session->num_images()) throw std::runtime_error("no frame to retry");
    const auto last = static_cast<std::int32_t>(session->num_images() - 1);
    result = step_json(session->retry(pairs_from(n_pairs, earlier, pair_begin, matches, last), new_piece != 0));
  } catch (const std::exception& e) {
    result = std::string("{\"ok\": false, \"error\": \"") + escape(e.what()) + "\"}";
  }
  return result.c_str();
}

// The map now: JSON (its cameras, intrinsics, pieces) and its points in demo_points()
// (3 floats each) and their tracks in demo_tracks().
EMSCRIPTEN_KEEPALIVE const char* demo_session_snapshot() {
  step_points.clear();
  step_tracks.clear();
  try {
    if (!session) throw std::runtime_error("no session");
    const demo::Snapshot s = session->snapshot();
    for (double v : s.points) step_points.push_back(static_cast<float>(v));
    step_tracks = s.point_track;
    std::ostringstream j;
    j << "{\"ok\": true, \"started\": " << (session->started() ? "true" : "false") << ", \"images\": " << session->num_images()
      << ", \"pieces\": " << session->num_pieces() << ", \"largest\": " << session->largest_piece()
      << ", \"points\": " << s.points.size() / 3 << ", \"mean_error_px\": ";
    num(j, s.mean_error_px);
    j << ", \"intrinsics\": [";
    const auto& in = session->intrinsics();
    for (std::size_t k = 0; k < in.size(); ++k) {
      if (k) j << ",";
      num(j, in[k]);
    }
    j << "], \"cameras\": ";
    write_cameras(j, *s.registered, *s.poses);
    j << "}";
    result = j.str();
  } catch (const std::exception& e) {
    result = std::string("{\"ok\": false, \"error\": \"") + escape(e.what()) + "\"}";
  }
  return result.c_str();
}

// The whole capture from the live map (Session::finish, Mapper::run_from): the session's
// pairs and more (each pair here with both of its frames, a and b, then the matches), with
// progress through Module.onProgress, which can stop it. The result as JSON, its points in
// demo_points(), their tracks in demo_tracks(), and per point an observation in
// demo_colour_obs().
EMSCRIPTEN_KEEPALIVE const char* demo_session_finish(int n_pairs, const std::int32_t* pair_images,
                                                    const std::int32_t* pair_begin, const std::int32_t* matches) {
  step_points.clear();
  step_tracks.clear();
  colour_obs.clear();
  try {
    if (!session) throw std::runtime_error("no session");
    demo::Capture c = session->capture();
    for (int p = 0; p < n_pairs; ++p) {
      demo::Pair pr;
      pr.a = pair_images[2 * p];
      pr.b = pair_images[2 * p + 1];
      for (std::int32_t m = pair_begin[p]; m < pair_begin[p + 1]; ++m) {
        pr.ka.push_back(matches[2 * m]);
        pr.kb.push_back(matches[2 * m + 1]);
      }
      c.pairs.push_back(std::move(pr));
    }
    const auto t0 = std::chrono::steady_clock::now();
    auto last = t0 - std::chrono::seconds(1);
    std::string last_stage;
    const demo::Reconstruction r = session->finish(c, [&](const demo::Progress& p) {
      const auto now = std::chrono::steady_clock::now();
      if (p.stage == last_stage && now - last < std::chrono::milliseconds(250)) return true;
      last = now;
      last_stage = p.stage;
      std::ostringstream j;
      j << "{\"stage\": \"" << p.stage << "\", \"registered\": " << p.registered << ", \"images\": " << p.images
        << ", \"iteration\": " << p.iteration << ", \"seconds\": ";
      num(j, std::chrono::duration<double>(now - t0).count());
      j << "}";
      return demo_report(j.str().c_str()) != 0;
    });
    for (double v : r.problem.points) step_points.push_back(static_cast<float>(v));
    step_tracks = r.point_track;
    colour_of(r);
    int registered = 0;
    for (char g : r.registered) registered += g ? 1 : 0;
    std::ostringstream j;
    j << "{\"ok\": " << (r.ok() ? "true" : "false") << ", \"cancelled\": " << (r.cancelled ? "true" : "false")
      << ", \"error\": \"" << escape(r.error) << "\", \"images\": " << c.num_images() << ", \"pairs\": " << c.pairs.size()
      << ", \"registered\": " << registered << ", \"points\": " << r.problem.points.size() / 3
      << ", \"observations\": " << r.problem.obs_point.size() << ", \"models\": " << r.models << ", \"mean_error_px\": ";
    num(j, r.mean_error_px);
    j << ", \"intrinsics\": [";
    for (std::size_t k = 0; k < r.problem.intrinsics.size(); ++k) {
      if (k) j << ",";
      num(j, r.problem.intrinsics[k]);
    }
    j << "], \"cameras\": ";
    write_cameras(j, r.registered, r.poses);
    j << ", \"seconds\": ";
    num(j, r.time.total);
    j << "}";
    result = j.str();
  } catch (const std::exception& e) {
    result = std::string("{\"ok\": false, \"error\": \"") + escape(e.what()) + "\"}";
  }
  return result.c_str();
}

EMSCRIPTEN_KEEPALIVE void demo_session_free() {
  session.reset();
  step_points.clear();
  step_tracks.clear();
  colour_obs.clear();
}

// A live frame's matches to a kept frame checked against a relative pose, on the CPU
// (Like Recon's capture loop: a camera guessed at 1.2 times the longer side, inliers
// within max_px). xy_a, xy_b: 2 floats per match, just-keypoints' convention; the size
// is both frames'. Writes one byte per match to mask (1: inlier) and returns the
// inliers, or -1 (the reason in the JSON of demo_last_error()).
EMSCRIPTEN_KEEPALIVE int demo_pair_inliers(int n, const float* xy_a, const float* xy_b, int width, int height,
                                          double max_px, std::uint8_t* mask) {
  if (n < 5) {
    for (int k = 0; k < n; ++k) mask[k] = 0;
    return 0;
  }
  if (!live_ctx) {
    jpo_config cfg;
    jpo_config_init(&cfg);
    cfg.backend = JPO_BACKEND_CPU;
    cfg.threads = 1;
    live_ctx = jpo_create(&cfg);
    if (!live_ctx) {
      result = std::string("{\"error\": \"") + escape(jpo_last_error()) + "\"}";
      return -1;
    }
  }
  std::vector<double> a(xy_a, xy_a + 2 * n), b(xy_b, xy_b + 2 * n);
  const std::uint32_t offsets[2] = {0, static_cast<std::uint32_t>(n)};
  jpo_camera cam{};
  cam.model = JPO_SIMPLE_PINHOLE;
  cam.params[0] = 1.2 * std::max(width, height);
  cam.params[1] = 0.5 * (width - 1);
  cam.params[2] = 0.5 * (height - 1);
  const std::uint32_t which = 0;
  jpo_pairs pairs{1, 1, offsets, a.data(), b.data(), &cam, &which, &which};
  jpo_ransac_options ransac;
  jpo_ransac_options_init(&ransac);
  ransac.max_error_px = max_px;
  jpo_pose_result r{};
  if (jpo_relative_pose(live_ctx, &pairs, &ransac, &r, mask) != 0) {
    result = std::string("{\"error\": \"") + escape(jpo_last_error()) + "\"}";
    return -1;
  }
  if (r.status != JPO_OK && r.status != JPO_DEGENERATE) {
    for (int k = 0; k < n; ++k) mask[k] = 0;
    return 0;
  }
  return static_cast<int>(r.n_inliers);
}

EMSCRIPTEN_KEEPALIVE const char* demo_last_error() { return result.c_str(); }

}  // extern "C"
