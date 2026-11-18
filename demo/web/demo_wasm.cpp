// The live demo's WebAssembly entry points: generate a scene, reconstruct it with the
// mapper, and return every step and the comparison with the ground truth as JSON.
// demo_run and demo_backends suspend on WebGPU (JSPI), so JavaScript calls them with
// ccall's {async: true}.
#include <emscripten/emscripten.h>

#include <chrono>
#include <cstdio>
#include <exception>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "evaluate.h"
#include "geometry.h"
#include "mapper.h"
#include "scenes.h"

namespace {

std::string result;  // the last answer; valid until the next call
std::unique_ptr<demo::Mapper> mapper;  // kept: on WebGPU its device start-up is paid once

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
  char buf[32];
  std::snprintf(buf, sizeof(buf), fmt, v);
  j << buf;
}

// Points and cameras are sent in the truth's frame: X_true = s R X + t.
struct Frame {
  double R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, t[3] = {0, 0, 0}, s = 1.0;
  void point(const double* X, double* Y) const {
    for (int d = 0; d < 3; ++d) Y[d] = s * (R[3 * d] * X[0] + R[3 * d + 1] * X[1] + R[3 * d + 2] * X[2]) + t[d];
  }
  // Camera centre and world-to-camera rotation of a pose, in the truth's frame.
  void camera(const double* pose, double C[3], double Rc[9]) const {
    double Cr[3], Re[9], Rt[9];
    demo::camera_centre(pose, Cr);
    point(Cr, C);
    demo::quat_to_rotmat(pose, Re);
    demo::transpose3(R, Rt);
    demo::mat3_mul(Re, Rt, Rc);
  }
};

void write_camera(std::ostringstream& j, const double C[3], const double R[9]) {
  j << "[";
  for (int k = 0; k < 3; ++k) {
    if (k) j << ",";
    num(j, C[k], "%.5g");
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
  std::vector<double> points;
  std::vector<std::int32_t> tracks;
  double mean_error_px = 0, seconds = 0;
};

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

// Reconstructs one generated scene. poses: "auto", "cpu" or "webgpu" (poselib-gpu);
// ba: "cpu" or "webgpu" (the solver).
EMSCRIPTEN_KEEPALIVE const char* demo_run(const char* scene_name, int seed, double noise_px, double wrong_fraction,
                                          int false_pairs, const char* poses, const char* ba) {
  try {
    demo::MapperOptions mo;
    const std::string pb = poses;
    mo.pose_backend = pb == "webgpu" ? JPO_BACKEND_WEBGPU : pb == "cpu" ? JPO_BACKEND_CPU : JPO_BACKEND_AUTO;
    const auto bb = jba::backend_from_name(ba);
    if (!bb) throw std::invalid_argument(std::string("unknown backend ") + ba);
    mo.ba_backend = *bb;
    mo.threads = 1;  // no threads in this build
    demo::thresholds_for_noise(noise_px, mo);
    if (!mapper || mapper->options().pose_backend != mo.pose_backend) {
      mapper.reset();
      mapper = std::make_unique<demo::Mapper>(mo);
    }
    mapper->set_options(mo);

    demo::SceneOptions so;
    so.seed = static_cast<std::uint64_t>(seed);
    so.noise_px = noise_px;
    so.wrong_fraction = wrong_fraction;
    so.false_pairs = false_pairs;
    const auto t_gen = std::chrono::steady_clock::now();
    const demo::Scene scene = demo::make_scene(scene_name, so);
    const double gen_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_gen).count();

    std::vector<Step> steps;
    const auto t0 = std::chrono::steady_clock::now();
    const demo::Reconstruction r = mapper->run(scene.capture, [&](const demo::Snapshot& s) {
      Step st;
      st.stage = s.stage;
      st.added = s.added;
      st.poses = *s.poses;
      st.registered = *s.registered;
      st.points = s.points;
      st.tracks = s.point_track;
      st.mean_error_px = s.mean_error_px;
      st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      steps.push_back(std::move(st));
    });
    const demo::Metrics mt = demo::evaluate(scene, r);
    Frame f;
    std::copy(mt.R, mt.R + 9, f.R);
    std::copy(mt.t, mt.t + 3, f.t);
    f.s = mt.s;

    std::size_t matches = 0;
    for (const demo::Pair& p : scene.capture.pairs) matches += p.ka.size();
    std::ostringstream j;
    j << "{\"ok\": " << (r.ok() ? "true" : "false") << ", \"error\": \"" << escape(r.error) << "\"";
    j << ", \"scene\": \"" << escape(scene.name) << "\", \"images\": " << scene.capture.num_images()
      << ", \"pairs\": " << scene.capture.pairs.size() << ", \"matches\": " << matches
      << ", \"model\": \"" << jba::model_name(scene.capture.model) << "\"";
    j << ", \"poses_backend\": \"" << escape(r.pose_backend) << "\", \"ba_backend\": \"" << jba::backend_name(mo.ba_backend)
      << "\"";
    j << ", \"truth_cameras\": [";
    for (std::size_t i = 0; i < scene.capture.num_images(); ++i) {
      double C[3], R[9];
      demo::camera_centre(&scene.poses[7 * i], C);
      demo::quat_to_rotmat(&scene.poses[7 * i], R);
      if (i) j << ",";
      write_camera(j, C, R);
    }
    j << "], \"steps\": [";
    for (std::size_t k = 0; k < steps.size(); ++k) {
      const Step& st = steps[k];
      j << (k ? "," : "") << "{\"stage\": \"" << st.stage << "\", \"seconds\": ";
      num(j, st.seconds);
      j << ", \"mean_error_px\": ";
      num(j, st.mean_error_px);
      j << ", \"added\": [";
      for (std::size_t a = 0; a < st.added.size(); ++a) j << (a ? "," : "") << st.added[a];
      j << "], \"cameras\": [";
      bool first = true;
      for (std::size_t i = 0; i < st.registered.size(); ++i) {
        if (!st.registered[i]) continue;
        double C[3], R[9];
        f.camera(&st.poses[7 * i], C, R);
        j << (first ? "" : ",") << "{\"image\": " << i << ", \"pose\": ";
        write_camera(j, C, R);
        j << "}";
        first = false;
      }
      j << "], \"tracks\": [";
      for (std::size_t p = 0; p < st.tracks.size(); ++p) j << (p ? "," : "") << st.tracks[p];
      j << "], \"points\": [";
      for (std::size_t p = 0; p < st.points.size() / 3; ++p) {
        double Y[3];
        f.point(&st.points[3 * p], Y);
        for (int d = 0; d < 3; ++d) {
          if (p || d) j << ",";
          num(j, Y[d], "%.4g");
        }
      }
      j << "]}";
    }
    j << "]";
    j << ", \"pairs_ok\": " << r.pairs_ok << ", \"pairs_degenerate\": " << r.pairs_degenerate
      << ", \"pairs_failed\": " << r.pairs_failed << ", \"pairs_kept\": " << r.pairs_kept
      << ", \"false_pairs_kept\": " << mt.false_pairs_kept << ", \"tracks\": " << r.tracks
      << ", \"first_pair\": [" << r.first_a << "," << r.first_b << "], \"first_angle_deg\": ";
    num(j, r.first_angle_deg);
    j << ", \"pnp_calls\": " << r.pnp_calls << ", \"pnp_rejected\": " << r.pnp_rejected
      << ", \"adjustments\": " << r.adjustments;
    j << ", \"metrics\": {\"registered\": " << mt.registered << ", \"points\": " << mt.points
      << ", \"observations\": " << mt.observations << ", \"mean_error_px\": ";
    num(j, mt.mean_error_px);
    j << ", \"rot_median_deg\": ";
    num(j, mt.rot_median_deg);
    j << ", \"rot_max_deg\": ";
    num(j, mt.rot_max_deg);
    j << ", \"centre_median_pct\": ";
    num(j, mt.centre_median_pct);
    j << ", \"point_median_pct\": ";
    num(j, mt.point_median_pct);
    j << ", \"mask_precision\": ";
    num(j, mt.mask_precision);
    j << ", \"mask_recall\": ";
    num(j, mt.mask_recall);
    j << ", \"pass\": " << (mt.pass ? "true" : "false") << ", \"why\": \"" << escape(mt.why) << "\"}";
    j << ", \"time\": {\"generate\": ";
    num(j, gen_seconds);
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
