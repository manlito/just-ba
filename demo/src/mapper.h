// A minimal incremental reconstruction (document 04, part C; D134). poselib-gpu
// verifies the matches of every pair, gives the first two poses and registers each
// further image by PnP; this library builds the tracks, triangulates and adjusts.
// Part of the demo app, not of the library: only the demo links poselib-gpu.
#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "jba/problem.h"
#include "jba/solver.h"
#include "jba/tracks.h"
#include "jpo/jpo.h"

namespace demo {

// The caller's matches of one image pair: keypoint ka[k] of image a with kb[k] of b.
struct Pair {
  std::int32_t a = 0, b = 0;
  std::vector<std::int32_t> ka, kb;
};

// Cameras of one model (normally one for the whole capture; calibrated, or refined by
// the mapper with refine_intrinsics), the keypoints
// of every image and the matches of the pairs the caller's matcher tried.
struct Capture {
  jba::CameraModel model = jba::CameraModel::Pinhole;
  std::vector<double> intrinsics;               // num_params(model) per block
  std::vector<std::int32_t> image_intrinsics;   // per image: its block; empty: block 0 for all
  std::vector<std::vector<double>> keypoints;  // per image: 2 per keypoint, pixels
  std::vector<Pair> pairs;
  // Per block: 1 when its focal length is only a guess (COLMAP's database without a
  // prior_focal_length); empty: all known.
  std::vector<std::uint8_t> focal_unknown;
  // Position priors (D150): where each image was taken, measured (GPS), as its camera
  // centre in a metric frame of the caller's whose third axis points up (east, north,
  // up about a point near the capture, for one). 3 per image, NaN for an image without
  // one; empty: none. prior_sigma: per image the standard deviations of the
  // measurement, horizontal then vertical, in the same units; empty, or NaN or not
  // positive for an image: MapperOptions' defaults.
  std::vector<double> prior_position;
  std::vector<double> prior_sigma;
  // Per block: its images' width and height in pixels (the capture file's camera.size);
  // empty: unknown, and MapperOptions::min_pair_coverage does not apply (D151).
  std::vector<std::int32_t> image_size;
  std::size_t num_images() const { return keypoints.size(); }
  std::int32_t block(std::size_t image) const { return image_intrinsics.empty() ? 0 : image_intrinsics[image]; }
  bool has_prior(std::size_t image) const {
    return !prior_position.empty() && std::isfinite(prior_position[3 * image]) &&
           std::isfinite(prior_position[3 * image + 1]) && std::isfinite(prior_position[3 * image + 2]);
  }
};

struct MapperOptions {
  std::int32_t pose_backend = JPO_BACKEND_AUTO;    // poselib-gpu's backend
  jba::BackendKind ba_backend = jba::BackendKind::Cpu;
  std::uint32_t seed = 7;
  int threads = 0;  // both libraries; 0: all

  // RANSAC inlier thresholds, pixels (poselib-gpu's defaults). Set from the keypoints'
  // noise: about twice its standard deviation for pairs.
  double pair_max_error_px = 1.0;
  double pnp_max_error_px = 4.0;

  // COLMAP's defaults where it has one.
  std::uint32_t min_pair_inliers = 30;     // a pair's inliers go to the tracks from this many
  double min_pair_inlier_ratio = 0.25;     // and this share of its matches
  // ...and when the convex hull of its inliers covers at least this share of each of its
  // two images (D151): matches bunched in one corner fix a relative pose poorly and are
  // the likeliest to come from a repeated or a false match. 0: no such rule; it needs
  // Capture::image_size.
  double min_pair_coverage = 0.0;
  std::uint32_t min_first_pair_inliers = 100;
  std::uint32_t min_pnp_inliers = 15;      // COLMAP: 30; the generated scenes are small
  double min_pnp_inlier_ratio = 0.25;
  double max_reproj_px = 4.0;              // the filter after each adjustment
  double min_tri_angle_deg = 1.5;
  // Largest angle between a keypoint's ray and its point for the keypoint to join the
  // point, at triangulation and when continuing a track, before the adjustment and its
  // reprojection filter. 0: max_reproj_px seen through the camera.
  double tri_max_angle_error_deg = 0.0;
  // Tracks with only two observations are not triangulated (COLMAP's mapper does the
  // same by default, Mapper.tri_ignore_two_view_tracks): they constrain little and are
  // the likeliest to be a wrong match.
  bool ignore_two_view_tracks = false;
  std::size_t pnp_batch = 4;               // images tried in one PnP call
  int local_iterations = 25;               // each adjustment after a registration
  int max_models = 10;                     // models tried, each from images in no earlier model
  int final_iterations = 50;
  // Cauchy scale of the adjustments, pixels; 0: max_reproj_px, so every observation the
  // filter accepts counts fully (D141). A smaller scale fits each point to the majority
  // of its keypoints and leaves valid ones, far from the image centre, to the filter.
  double loss_scale = 0.0;
  // Intrinsics groups the adjustments refine (a jba::IntrinsicsGroup mask), as COLMAP's
  // mapper refines focal length and distortion (Mapper.ba_refine_*); 0: all fixed, for
  // a calibrated capture. A block is refined once at least refine_min_images registered
  // images use it; the refined values then serve triangulation and PnP too.
  std::uint8_t refine_intrinsics = 0;
  int refine_min_images = 1;
  // Before anything else, the focal lengths of the blocks marked focal_unknown are
  // searched for: the pairs are verified under focal lengths from a quarter to four
  // times the guess (61 steps of 4.7 %), all in one poselib-gpu call, and the factor
  // with the lowest total MSAC score is applied. The adjustments then refine it.
  bool search_focal = false;
  // When no image sees min_pnp_inliers of the map's points, the image with the most
  // inliers to a registered one, in a kept pair with at least min_register_pair_inliers
  // of them, is registered from that pair instead: its rotation the pair's, its centre
  // along the pair's translation at the map's median spacing of neighbouring cameras;
  // its points are triangulated and the map adjusted as after a PnP (D158). After a
  // turn, a capture's next frames match their neighbours by the hundreds yet see almost
  // none of the map's points (those of the turn, seen only from nearly one place, are
  // not triangulated), so PnP cannot place them and the model ends there. false: COLMAP's
  // behaviour, a new model.
  bool register_from_pairs = false;
  std::uint32_t min_register_pair_inliers = 100;

  // Position priors (Capture::prior_position, D150). With them, a model starts from a
  // pair of images that both have one where such a pair is fit to start from; once
  // prior_min_images of its registered images have one, spread across their line by
  // more than prior_min_spread standard deviations, the model is moved into the
  // priors' frame by the similarity of its camera centres onto their priors (robust:
  // centres within prior_inlier_sigmas of their prior), and from then on every
  // adjustment holds each registered image with a prior to it (a global position prior
  // of the working frame, Cauchy at 3 standard deviations). The model then grows in the
  // priors' frame, units and axes, and so does every other model the run grows.
  // false: priors are ignored.
  bool use_position_priors = true;
  int prior_min_images = 3;
  double prior_min_spread = 5.0;
  double prior_inlier_sigmas = 4.0;
  // The standard deviations for images whose prior comes without them: an ordinary
  // GPS receiver's (a drone's or a phone's), horizontal and vertical.
  double prior_sigma_horizontal = 3.0;
  double prior_sigma_vertical = 5.0;
};

// The median focal length over the intrinsics blocks, pixels (fx and fy averaged).
double median_focal(jba::CameraModel model, const std::vector<double>& intrinsics);

// Thresholds for keypoints whose noise is sigma pixels per coordinate. The defaults hold
// up to half a pixel; above that the thresholds grow with it.
inline void thresholds_for_noise(double sigma, MapperOptions& o) {
  o.pair_max_error_px = sigma > 0.5 ? 2.0 * sigma : 1.0;
  o.pnp_max_error_px = sigma > 1.0 ? 4.0 * sigma : 4.0;
  o.max_reproj_px = sigma > 1.0 ? 4.0 * sigma : 4.0;
}

// The defaults of COLMAP's incremental mapper and two-view geometry, for real captures:
// RANSAC at 4 px for pairs and 12 px for PnP, at least 30 PnP inliers, filter at 4 px.
// Long-standing COLMAP defaults; not checked against 4.2.1 on this machine.
inline void colmap_thresholds(MapperOptions& o) {
  o.pair_max_error_px = 4.0;
  o.min_pair_inliers = 15;  // COLMAP's two-view geometry keeps pairs from 15 inliers
  o.pnp_max_error_px = 12.0;
  o.min_pnp_inliers = 30;
  o.min_pnp_inlier_ratio = 0.25;
  o.min_first_pair_inliers = 100;
  o.max_reproj_px = 4.0;
  o.min_tri_angle_deg = 1.5;
  o.ignore_two_view_tracks = true;
  o.tri_max_angle_error_deg = 2.0;  // COLMAP's create_max_angle_error and continue_max_angle_error
  o.refine_intrinsics = jba::kFocal | jba::kDistortion;  // principal point fixed, as COLMAP's
  // Not COLMAP's: it estimates a guessed focal length from the fundamental matrix and
  // in PnP, which poselib-gpu has no solver for (D144).
  o.search_focal = true;
}

// The map after a step, for a caller that draws the reconstruction as it grows.
struct Snapshot {
  std::string stage;                       // "first pair", "register", "final"
  std::vector<std::int32_t> added;         // images registered in this step
  const std::vector<double>* poses;        // 7 per image; meaningful where registered
  const std::vector<char>* registered;
  std::vector<double> points;              // 3 per triangulated point
  std::vector<std::int32_t> point_track;   // per point: its track, the same across steps
  double mean_error_px = 0.0;
  bool aligned = false;                    // in the frame of the position priors (D150)
};
using StepCallback = std::function<void(const Snapshot&)>;

// Where a run is, reported as it happens (D146), for a caller that shows live progress
// and offers to stop. Stages, in order: "focal search" (only with search_focal),
// "verify", "tracks"; then per model "first pair", "register" (after each registration's
// adjustment) and "final", each with `live`; and "adjust" after every solver iteration
// of every adjustment, with `iteration` and `cost`. Unlike on_step's snapshots, which
// come after the run and only for the model kept, live snapshots are of the model being
// grown, which may not be the one returned.
struct Progress {
  std::string stage;
  int model = -1;        // the model being grown, from 0; -1 before the first
  int registered = 0;    // images registered in it
  int images = 0;        // in the capture
  int adjustments = 0;   // finished so far, every model
  int iteration = -1;    // "adjust": the solver's iteration, from 0; -1 otherwise
  double cost = 0.0;     // "adjust": the cost after it
  const Snapshot* live = nullptr;  // "first pair", "register", "final"; valid during the call only
};
// Return false to stop: the run ends as soon as it can, with Reconstruction::cancelled.
using ProgressCallback = std::function<bool(const Progress&)>;

struct Timings {
  double focal_search = 0, verify = 0, tracks = 0, pnp = 0, triangulate = 0, adjust = 0, total = 0;  // seconds
};

struct Reconstruction {
  std::string error;  // empty on success
  bool ok() const { return error.empty(); }
  // The progress callback returned false; error is then "cancelled" and the rest of
  // the result is incomplete.
  bool cancelled = false;
  double focal_factor = 1.0;  // the search's factor (search_focal), 1 without one

  std::vector<double> poses;        // 7 per image of the capture
  std::vector<char> registered;
  // The last adjustment: registered images and their points, observations kept by
  // the filter. image_of maps its images to the capture's; obs_keypoint gives each
  // observation's keypoint in its image.
  jba::Problem problem;
  std::vector<std::int32_t> image_of;
  std::vector<std::int32_t> obs_keypoint;
  std::vector<std::int32_t> point_track;  // per point of `problem`: its track in `all_tracks`
  jba::Tracks all_tracks;                  // every track built from the verified matches
  jba::TrackStats track_stats;             // of building them
  double mean_error_px = 0.0;

  // Position priors (D150), of the model returned. aligned: it is in the priors' frame
  // (D150's alignment happened); otherwise its frame is the first pair's. prior_images:
  // its registered images with a prior; prior_inliers: those within
  // prior_inlier_sigmas of their prior after the last adjustment. The errors are the
  // medians, over prior_images, of the distance from each camera centre to its prior,
  // horizontally and vertically, in the priors' units (0 unless aligned).
  bool aligned = false;
  int prior_images = 0, prior_inliers = 0;
  double prior_error_horizontal = 0.0, prior_error_vertical = 0.0;

  // Pair verification.
  std::vector<jpo_pose_result> pair_results;
  std::vector<std::uint8_t> pair_inliers;  // one byte per match, every pair in order
  std::vector<std::uint32_t> pair_offsets;
  std::vector<char> pair_kept;
  int pairs_ok = 0, pairs_degenerate = 0, pairs_failed = 0, pairs_kept = 0;
  // Per pair: the share of its images the convex hull of its inliers covers, the smaller
  // of the two (D151); NaN without Capture::image_size. pairs_narrow: pairs with enough
  // inliers left out by min_pair_coverage.
  std::vector<float> pair_coverage;
  int pairs_narrow = 0;

  // Every model grown, in the order grown (D151); `kept_model` indexes the one returned.
  // The others are what a capture that falls apart is made of: each in its own frame
  // (the first pair's, or the priors' when it was aligned), not merged with the rest.
  struct Model {
    std::vector<char> registered;  // per image of the capture
    std::vector<double> poses;     // 7 per image of the capture; meaningful where registered
    std::vector<double> points;    // 3 per point of its last adjustment
    std::size_t observations = 0;
    double mean_error_px = 0.0;
    int first_pair = -1;           // index into Capture::pairs
    bool aligned = false;
  };
  std::vector<Model> all_models;
  int kept_model = -1;
  // Per image of the capture, in the model returned (D151): the distinct points of it the
  // image sees, the inliers of its last PnP that was rejected (-1: none was), and the
  // mean reprojection error of its observations after the last adjustment (NaN where
  // not registered). Why an image was left out follows from them and the pairs: in no
  // kept pair; fewer points seen than min_pnp_inliers; or a pose rejected.
  std::vector<std::int32_t> image_points_seen;
  std::vector<std::int32_t> image_pnp_inliers;
  std::vector<double> image_error_px;

  std::size_t tracks = 0;  // built from the matches, before any split
  int first_a = -1, first_b = -1;
  std::uint32_t first_inliers = 0;
  double first_angle_deg = 0.0;
  int pnp_calls = 0, pnp_problems = 0, pnp_rejected = 0, adjustments = 0;
  int pair_registrations = 0;  // images registered from a pair (D158)
  int start_left_out = 0;             // run_from: start images that disagreed with their pairs
  int models = 0;                     // models grown; the one returned has the most images
  std::string pose_backend;           // what poselib-gpu ran on
  Timings time;
};

// Where a run starts instead of a first pair (D153): a model already grown, normally a
// Session's live map. Per image of the capture its registered flag and pose, and the
// intrinsics, blocks as the capture's.
struct StartFrom {
  std::vector<char> registered;
  std::vector<double> poses;       // 7 per image
  std::vector<double> intrinsics;
  bool aligned = false;            // in the frame of the position priors (D150)
  double focal_factor = 1.0;       // reported as the run's
};

// How well a result's cameras agree with its verified pairs (D153): per kept pair whose
// two images are registered, the angle between the relative rotation of their poses and
// the pair's own (poselib-gpu's, from its matches), in degrees; the `q` quantile of
// those (0.5 the median), or NaN without such pairs.
double pair_rotation_error_deg(const Capture& capture, const Reconstruction& r, double q);

// One poselib-gpu context, kept across reconstructions (on a GPU its start-up is
// paid once).
class Mapper {
 public:
  explicit Mapper(const MapperOptions& options);  // throws std::runtime_error
  ~Mapper();
  Mapper(const Mapper&) = delete;
  Mapper& operator=(const Mapper&) = delete;

  // on_step: the kept model's steps, after the run. on_progress: live, and can stop it.
  // Neither changes the result.
  Reconstruction run(const Capture& capture, const StepCallback& on_step = {},
                     const ProgressCallback& on_progress = {});
  // The same run with its first model taken from `start` (D153): no focal search (the
  // start's intrinsics are used), every pair verified, the tracks built; the start's
  // registered images keep their poses (but those whose relative rotations disagree
  // with their verified pairs by a median of over 5 degrees, which are left out and
  // registered again like any other), their points are triangulated and everything is
  // adjusted (stage "resume", final_iterations), then images are registered and the
  // model finished as in `run`, and further models grown from images in none. With
  // fewer than two start images left, the run is `run`'s.
  Reconstruction run_from(const Capture& capture, const StartFrom& start, const StepCallback& on_step = {},
                          const ProgressCallback& on_progress = {});
  const MapperOptions& options() const { return options_; }
  // Options for the next runs. The context keeps the pose backend, seed and threads it
  // was created with; everything else, the solver's backend included, can change.
  void set_options(const MapperOptions& options) { options_ = options; }

 private:
  friend class Session;
  Reconstruction run_impl(const Capture& capture, const StartFrom* start, const StepCallback& on_step,
                          const ProgressCallback& on_progress);
  MapperOptions options_;
  jpo_ctx* ctx_ = nullptr;
};

// ---- Sessions (D153): the map grown one image at a time, while a capture is taken.

struct SessionOptions {
  // Images adjusted together after each registration: those just registered and the
  // most recent before them, up to this many; the other registered images that see
  // their points are held. 0: every registered image, every time.
  int window = 8;
  int window_iterations = 0;  // 0: MapperOptions::local_iterations
  // The map starts from a pair whose median triangulation angle reaches 16 degrees
  // (COLMAP's first floor); after start_patience images without one, 8; after twice as
  // many, 4; then 2 (COLMAP's floors), never below min_start_angle_deg. A pair with a
  // small angle makes a map that bends; waiting costs only the delay, as the images
  // that came meanwhile are placed when it starts.
  int start_patience = 4;
  double min_start_angle_deg = 4.0;
  // A piece started from a pair under 16 degrees is started again, from a pair of its
  // images (or of images in no piece) that reaches a higher floor, if one comes while
  // it has at most this many images. 0: never.
  int rebuild_within = 16;
  // When more than this many images in a row could not be placed, the map is put aside
  // as a piece and a new one started from a pair of images in no piece, if one is fit
  // to start from (a capture that lost its way). 0: never. An image the current piece
  // can't place but whose pairs link to an earlier piece goes to that piece, which
  // becomes the current one again (the capture came back).
  int restart_after = 2;
};

// An image for Session::add. Its pairs name an earlier image of the session in `a` and
// this one in `b` (add sets it): ka are the earlier image's keypoints, kb this one's.
struct SessionImage {
  std::vector<double> keypoints;  // 2 per keypoint, pixels
  std::vector<Pair> pairs;
  // Optional priors (D150): where the image was taken, measured (a phone's tracker, a
  // GPS), and the standard deviations, horizontal and vertical; NaN for none, or for
  // MapperOptions' defaults. Only the camera centre is used for now.
  double prior_position[3] = {std::nan(""), std::nan(""), std::nan("")};
  double prior_sigma[2] = {std::nan(""), std::nan("")};
  // Not placed, the caller will try more pairs (Session::retry) before a new piece may
  // start: the restart rule waits for retry.
  bool defer_split = false;
};

// A model a session starts from (D154): images already placed, with their keypoints,
// poses, and the model's points with the keypoints that observe them (a finished
// reconstruction, so a capture can go on into it). Its images are held where they are
// and its intrinsics are kept while new images join; its points take part in placing
// them. Per observation: its image (from 0), its keypoint in that image, its point.
struct SessionSeed {
  std::vector<std::vector<double>> keypoints;  // per image: 2 per keypoint, pixels
  std::vector<double> poses;                   // 7 per image
  std::vector<double> points;                  // 3 per point
  std::vector<std::int32_t> obs_image, obs_keypoint, obs_point;
  bool aligned = false;                        // in the frame of the position priors (D150)
};

// What Session::add did.
struct SessionStep {
  std::int32_t image = -1;           // the image added, from 0
  bool started = false;              // the map (or a new piece of it) started with this image
  int piece = -1;                    // the piece being grown, from 0; -1 before the map starts
  bool rebuilt = false;              // the piece was started again from a wider pair (started too)
  bool rejoined = false;             // an earlier piece took the image and is the current one now
  std::vector<std::int32_t> placed;  // registered in this step: this image, and earlier ones it lets in
  std::vector<std::int32_t> window;  // adjusted in this step, poses free
  std::vector<std::int32_t> held;    // in that adjustment, poses held
  int pairs_kept = 0;                // of this image's pairs, verified and kept for the tracks
  int pnp_inliers = -1;              // of this image's last PnP; -1: none tried
  int points_seen = 0;               // distinct points of the map this image sees
  double mean_error_px = 0.0;        // after the last adjustment
  double seconds = 0.0;
  std::string error;                 // empty on success
  bool ok() const { return error.empty(); }
};

// A map that grows image by image (D153). Each image comes with its matches to earlier
// images: they are verified, added to the tracks, and once the map has started (the
// first pair fit to start from, after a focal search when the focal length is a guess)
// the image is placed by PnP, its new points triangulated and a window of the latest
// images adjusted with the older ones held. Rejected images are tried again as the map
// grows. finish() then runs the mapper over the whole capture from the live map.
class Session {
 public:
  // The capture's camera: one intrinsics block of `model` (its focal length only a
  // guess, or not), its images' size. The mapper is used for its options and context,
  // and must outlive the session.
  Session(Mapper& mapper, jba::CameraModel model, const std::vector<double>& intrinsics, bool focal_unknown,
          std::int32_t width, std::int32_t height, const SessionOptions& options = {});
  // A session started from a model (D154): `seed`'s images are the session's first,
  // placed and held, the map started from them; `intrinsics` are the model's (its block
  // of `model`, not searched, held while the session grows). Throws
  // std::invalid_argument when the seed's arrays disagree or name what doesn't exist.
  Session(Mapper& mapper, jba::CameraModel model, const std::vector<double>& intrinsics, std::int32_t width,
          std::int32_t height, const SessionSeed& seed, const SessionOptions& options = {});
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  Session(Session&&) noexcept;
  Session& operator=(Session&&) noexcept;

  SessionStep add(const SessionImage& image);
  // The last image added again, with more pairs (to earlier images, a: the earlier
  // one), when it wasn't placed: they are verified and tracked, the image placed if it
  // now can be (into an earlier piece too), and then the restart rule applies, or a
  // new piece starts at once when `new_piece` (a caller's Shoot now).
  SessionStep retry(const std::vector<Pair>& more, bool new_piece = false);

  std::size_t num_images() const;
  // Images the session started with (D154): the first num_seeded() of it; 0 without.
  std::size_t num_seeded() const;
  bool started() const;
  // Pieces grown so far, the current one included; earlier pieces are kept as they
  // were left (piece(k), k below num_pieces() - 1, the current one is snapshot()).
  std::size_t num_pieces() const;
  Snapshot piece(std::size_t k) const;
  // The piece with the most images placed (the current one on a tie), its
  // intrinsics, and per keypoint of image `i` the track whose point it observes in
  // piece `k` (-1 for none).
  std::size_t largest_piece() const;
  const std::vector<double>& piece_intrinsics(std::size_t k) const;
  std::vector<std::int32_t> keypoint_points(std::size_t i, std::size_t k) const;
  // The map now, stage "session": every point with its track (track indices stay the
  // same as the map grows). The pointers are valid until the next add.
  Snapshot snapshot() const;
  // The map's intrinsics (refined as it grows).
  const std::vector<double>& intrinsics() const;
  // Per keypoint of image `i`: the track whose point it observes, -1 for none.
  std::vector<std::int32_t> keypoint_points(std::size_t i) const;
  // Everything added: the keypoints, the pairs as given, the priors.
  const Capture& capture() const;
  // The later run (D153): `capture`, the session's images with any pairs (normally the
  // session's matched again more thoroughly and pairs it did not have), reconstructed
  // with Mapper::run_from from the live map, or from the piece with the most images
  // when an earlier one has more. Before the map started, Mapper::run.
  Reconstruction finish(const Capture& capture, const ProgressCallback& on_progress = {});

 private:
  struct State;
  std::unique_ptr<State> s_;
};

}  // namespace demo
