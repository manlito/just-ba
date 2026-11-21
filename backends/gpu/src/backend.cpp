#include "jba/gpu/backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "jba/cpu/reprojection.h"
#include "jba/cpu/system.h"
#include "jba/geometry.h"
#include "jba/loss.h"

namespace jba::gpu {
namespace {

using jba_gpu::Chunk;
using jba_gpu::Factor;
using jba_gpu::FBlockDesc;
using jba_gpu::kChunkFloats;
using jba_gpu::kGatherThreads;
using jba_gpu::kReduceThreads;
using jba_gpu::kTileThreads;
using jba_gpu::Tile;
using jba_gpu::TileChunk;
using jba_gpu::TileEntry;
using jba_gpu::uint_pair;

static_assert(static_cast<int>(LossType::None) == jba_gpu::kNone && static_cast<int>(LossType::Huber) == jba_gpu::kHuber &&
              static_cast<int>(LossType::Cauchy) == jba_gpu::kCauchy &&
              static_cast<int>(LossType::GemanMcClure) == jba_gpu::kGemanMcClure);
static_assert(static_cast<int>(cpu::BlockKind::Pose) == jba_gpu::kPose &&
              static_cast<int>(cpu::BlockKind::Intrinsics) == jba_gpu::kIntrinsics &&
              static_cast<int>(cpu::BlockKind::Depth) == jba_gpu::kDepth &&
              static_cast<int>(cpu::BlockKind::IntrinsicsGroup) == jba_gpu::kIntrinsicsGroup);

// A predicted decrease smaller than this share of the sum of its terms' magnitudes
// cannot be resolved by the single-precision sums that form it.
constexpr double kResolvable = 1e-5;

// Cholesky pivots of the point and camera blocks below this share of their diagonal
// entry are raised to it (solver.inc, spd_inverse): about eight single-precision
// epsilons, the rounding error of a pivot itself, so a pivot below it carries no
// information. Measured on BAL ladybug-49: 1e-7 lets points with a degenerate depth
// direction take steps of 1e5; 1e-5 and above slow convergence (damping where the
// block is in fact resolved).
constexpr float kPivotFloor = 1e-6f;

constexpr std::size_t kChunkObservations = 32;
constexpr std::size_t kTileChunk = 1024;  // entries per group of schur_tiles
constexpr int kResidualResetPeriod = 10;  // Ceres's conjugate gradients default
constexpr int kMaxFailedFactorizations = 3;  // in a row, for a reason other than the matrix
// Test hooks (testing::).
int testing_factor_failures = 0;
FactorStatus testing_factor_status = FactorStatus::NotPositiveDefinite;
bool testing_fail_setup = false;

Kernel linearize_kernel(CameraModel model) {
  switch (model) {
    case CameraModel::SimplePinhole: return Kernel::LinearizeSimplePinhole;
    case CameraModel::Pinhole: return Kernel::LinearizePinhole;
    case CameraModel::SimpleRadial: return Kernel::LinearizeSimpleRadial;
    case CameraModel::Radial: return Kernel::LinearizeRadial;
    case CameraModel::OpenCV: return Kernel::LinearizeOpenCV;
    case CameraModel::OpenCVFisheye: return Kernel::LinearizeOpenCVFisheye;
  }
  return Kernel::LinearizePinhole;
}

}  // namespace

struct GpuBackend::Impl {
  cpu::System sys;
  std::unique_ptr<Device> dev;
  Buffer state_buf;
  Kernel linearize_k = Kernel::LinearizePinhole;

  // Buffers the host writes or reads.
  Buffer poses, rc, points, intr, basis, f_scale, e_scale, hres;
  Buffer dr, dJp, dJx, dJc, fr, fJ;
  Buffer diag_f, g_f, diag_e, g_e, x, dx_e, bvec;
  Args args{};

  double centre[3] = {0, 0, 0};
  bool any_intr = false;
  std::size_t nchunks = 0, nfac = 0, npts = 0, ngunits = 0, ngsplits = 0;
  std::vector<std::size_t> set_joff, set_roff;  // per factor set, into fJ and fr
  double gmax = 0;
  // Whitened reprojection residuals in double rounded to float, at the current and
  // the candidate parameters: by-products of their costs.
  std::vector<float> res_cur, res_cand;
  bool res_cur_valid = false;

  // Strategy B: the reduced camera system's tiles on the device, its lower triangle in
  // compressed columns for the direct solver (csc_src: each value's place among the
  // tiles'), analyzed once.
  bool direct = false;
  bool allow_fallback = false;  // Auto: strategy B may give way to A (D102)
  std::string note;             // why it did
  int fallback_steps = 0;       // sparse_schur steps solved by conjugate gradients instead
  int failed_in_a_row = 0;      // factorizations that failed for a reason other than the matrix
  std::vector<int> block_at;    // f offset -> camera-side block
  std::vector<std::pair<const cpu::FactorSet*, std::size_t>> fac_index;  // factor -> (set, index in it)
  std::size_t ntiles = 0, ntile_chunks = 0;
  Buffer tile_vals;
  std::vector<std::int64_t> csc_begin;
  std::vector<int> csc_rows;
  std::vector<std::uint32_t> csc_src;
  std::vector<double> csc_vals, rhs;
  std::unique_ptr<DirectSolver> solver;

  // try_direct: Auto may take strategy B (false on the second attempt, D102).
  Impl(const Problem& p, const SolverOptions& o, const Gauge& gauge, std::unique_ptr<Device> device, bool try_direct)
      : sys(p, o, gauge), dev(std::move(device)), allow_fallback(o.linear_solver == LinearSolverType::Auto) {
    if (o.linear_solver == LinearSolverType::SparseSchur || (allow_fallback && try_direct)) {
      solver = dev->direct_solver();
      if (solver) direct = true;
      else if (!allow_fallback)
        throw std::runtime_error(dev->name() + ": sparse_schur is not available on this device");
    }
    dev->scoped([&] { setup(p); });
  }

  // A buffer of `count` T, copied from init or zeroed; its device address in `field`.
  template <typename T, typename F>
  Buffer buffer(std::size_t count, F*& field, const T* init = nullptr, bool host_visible = false) {
    static_assert(sizeof(T) == sizeof(F));
    // Devices take at least 16 bytes and copy that many from init: a shorter array is
    // padded with zeros first (review of milestone 5, finding 2).
    const std::size_t bytes = count * sizeof(T);
    unsigned char pad[16] = {};
    const void* src = count > 0 ? init : nullptr;
    if (src != nullptr && bytes < sizeof(pad)) {
      std::memcpy(pad, src, bytes);
      src = pad;
    }
    const Buffer b = dev->alloc(std::max<std::size_t>(bytes, sizeof(pad)), src, host_visible);
    field = reinterpret_cast<F*>(static_cast<std::uintptr_t>(b.address));
    return b;
  }

  // A buffer the host reads or writes.
  template <typename T, typename F>
  Buffer mirrored(std::size_t count, F*& field) {
    return buffer<T>(count, field, nullptr, true);
  }

  template <typename T, typename S, typename F>
  Buffer converted(const std::vector<S>& v, F*& field) {
    std::vector<T> c(v.begin(), v.end());
    return buffer<T>(c.size(), field, c.data());
  }

  static float* host(const Buffer& b) { return static_cast<float*>(b.host); }
  State& state() { return *static_cast<State*>(state_buf.host); }

  void setup(const Problem& p) {
    const std::size_t nobs = sys.nobs, n_img = p.num_images(), n_blk = p.num_intrinsics_blocks();
    npts = sys.point_off.size();
    const std::size_t np = static_cast<std::size_t>(sys.np);
    any_intr = std::any_of(sys.intr_off.begin(), sys.intr_off.end(), [](int v) { return v >= 0; });

    // Centre of the scene (document 03): the points' mean, else the cameras'.
    std::size_t count = 0;
    for (std::size_t j = 0; j < npts; ++j) {
      if (sys.pt_begin[j] == sys.pt_begin[j + 1]) continue;
      for (int c = 0; c < 3; ++c) centre[c] += sys.cur.points[3 * j + c];
      ++count;
    }
    if (count == 0) {
      for (std::size_t i = 0; i < n_img; ++i) {
        const double* pose = &sys.cur.poses[7 * i];
        const double conj[4] = {pose[0], -pose[1], -pose[2], -pose[3]};
        double C[3];
        quat_rotate(conj, pose + 4, C);
        for (int c = 0; c < 3; ++c) centre[c] -= C[c];
        ++count;
      }
    }
    if (count > 0)
      for (double& v : centre) v /= static_cast<double>(count);

    // Observations.
    converted<std::int32_t>(sys.o_img, args.o_img);
    converted<std::int32_t>(sys.o_pt, args.o_pt);
    converted<float>(sys.o_xy, args.o_xy);
    converted<float>(sys.o_w, args.o_w);
    converted<std::int32_t>(sys.o_pose, args.o_pose);
    converted<std::int32_t>(sys.o_intr, args.o_intr);
    if (sys.has_depth) {
      converted<float>(sys.o_dw, args.o_dw);
      converted<std::int32_t>(sys.o_dep, args.o_dep);
    } else {
      buffer<float>(1, args.o_dw);
      buffer<std::int32_t>(1, args.o_dep);
    }
    converted<std::uint32_t>(sys.pt_begin, args.pt_begin);
    converted<std::int32_t>(sys.point_off, args.point_off);
    converted<std::int32_t>(sys.image_block, args.image_block);
    converted<std::int32_t>(sys.pose_off, args.pose_off);
    converted<std::int32_t>(sys.intr_off, args.intr_off);
    std::vector<std::uint8_t> pmask(n_img);
    for (std::size_t i = 0; i < n_img; ++i) pmask[i] = sys.pose_off[i] < 0 ? 0x3f : sys.pose_fixed_cols[i];
    buffer<std::uint8_t>(pmask.size(), args.pose_mask, pmask.data());
    std::vector<std::uint16_t> kmask(n_blk);
    for (std::size_t c = 0; c < n_blk; ++c) kmask[c] = sys.intr_off[c] < 0 ? 0xffff : sys.intr_fixed_cols[c];
    buffer<std::uint16_t>(kmask.size(), args.intr_mask, kmask.data());
    converted<std::int32_t>(sys.basis_of, args.basis_of);
    converted<std::uint32_t>(sys.img_list, args.img_list);
    converted<std::uint32_t>(sys.ib_list, args.ib_list);

    // Camera-side blocks and their chunks: runs of at most kChunkObservations
    // observations, extended to the end of a point's observations.
    std::vector<FBlockDesc> fbd;
    std::vector<Chunk> chunks;
    std::vector<std::uint32_t> block_chunks{0};
    for (std::size_t b = 0; b < sys.fblocks.size(); ++b) {
      const auto& fb = sys.fblocks[b];
      fbd.push_back({fb.offset, fb.size, fb.pinv_offset, static_cast<std::int32_t>(fb.kind)});
      std::size_t q0, q1;
      const std::vector<std::size_t>* list;
      sys.obs_range(fb, q0, q1, list);
      const std::uint32_t which = fb.kind == cpu::BlockKind::Intrinsics ? 1 : 0;
      for (std::size_t q = q0; q < q1;) {
        std::size_t end = std::min(q1, q + kChunkObservations);
        while (end < q1 && sys.o_pt[(*list)[end]] == sys.o_pt[(*list)[end - 1]]) ++end;
        chunks.push_back({static_cast<std::uint32_t>(b), which, static_cast<std::uint32_t>(q),
                          static_cast<std::uint32_t>(end)});
        q = end;
      }
      block_chunks.push_back(static_cast<std::uint32_t>(chunks.size()));
    }
    nchunks = chunks.size();
    buffer<FBlockDesc>(fbd.size(), args.fblocks, fbd.data());
    buffer<Chunk>(chunks.size(), args.chunks, chunks.data());
    buffer<std::uint32_t>(block_chunks.size(), args.block_chunks, block_chunks.data());

    // Gather units: a group of block_gather per block, or per run of at most
    // kGatherUnit observations where a block has more (a shared intrinsics block sees
    // every observation, and one group would take it alone); a split block's units
    // are added in order by gather_finish.
    std::vector<jba_gpu::GatherUnit> gunits;
    std::vector<jba_gpu::GatherSplit> gsplits;
    std::uint32_t nslots = 0;
    for (std::size_t b = 0; b < sys.fblocks.size(); ++b) {
      std::size_t q0, q1;
      const std::vector<std::size_t>* list;
      sys.obs_range(sys.fblocks[b], q0, q1, list);
      const auto ub = static_cast<std::uint32_t>(b);
      if (q1 - q0 <= jba_gpu::kGatherUnit) {
        gunits.push_back({ub, static_cast<std::uint32_t>(q0), static_cast<std::uint32_t>(q1), jba_gpu::kNoSlot});
        continue;
      }
      const std::uint32_t s0 = nslots;
      for (std::size_t q = q0; q < q1; q += jba_gpu::kGatherUnit)
        gunits.push_back({ub, static_cast<std::uint32_t>(q),
                          static_cast<std::uint32_t>(std::min(q1, q + jba_gpu::kGatherUnit)), nslots++});
      gsplits.push_back({ub, s0, nslots, 0});
    }
    ngunits = gunits.size();
    ngsplits = gsplits.size();
    buffer<jba_gpu::GatherUnit>(gunits.size(), args.gunits, gunits.data());
    if (!gsplits.empty()) {
      buffer<jba_gpu::GatherSplit>(gsplits.size(), args.gsplits, gsplits.data());
      buffer<float>(16 * static_cast<std::size_t>(nslots), args.gpartial);
    } else {
      buffer<jba_gpu::GatherSplit>(1, args.gsplits);  // never read
      buffer<float>(16, args.gpartial);
    }

    // Factors, flattened over the sets, and each block's (factor, slot) list in the
    // sets' order.
    std::vector<Factor> facs;
    std::vector<std::size_t> set_first;
    std::size_t joff = 0, roff = 0;
    for (const auto& fs : sys.fsets) {
      set_first.push_back(facs.size());
      set_joff.push_back(joff);
      set_roff.push_back(roff);
      for (std::size_t f = 0; f < fs.count; ++f) {
        Factor F{};
        F.rows = fs.rows;
        F.cols = fs.cols;
        F.joff = static_cast<std::int32_t>(joff + static_cast<std::size_t>(fs.rows * fs.cols) * f);
        F.roff = static_cast<std::int32_t>(roff + static_cast<std::size_t>(fs.rows) * f);
        F.nslots = fs.nslots;
        for (int s = 0; s < 3; ++s) {
          F.off[s] = s < fs.nslots ? fs.off(f, s) : -1;
          F.base[s] = fs.slot_base[s];
          F.size[s] = fs.slot_size[s];
        }
        facs.push_back(F);
      }
      joff += static_cast<std::size_t>(fs.rows * fs.cols) * fs.count;
      roff += static_cast<std::size_t>(fs.rows) * fs.count;
    }
    nfac = facs.size();
    std::vector<std::uint32_t> fb_fac_begin{0};
    std::vector<uint_pair> fb_fac;
    for (std::size_t b = 0; b < sys.fblocks.size(); ++b) {
      for (std::size_t s = 0; s < sys.fsets.size(); ++s) {
        const auto& fs = sys.fsets[s];
        for (std::size_t q = fs.begin[b]; q < fs.begin[b + 1]; ++q)
          fb_fac.push_back({static_cast<std::uint32_t>(set_first[s] + static_cast<std::size_t>(fs.list[q][0])),
                            static_cast<std::uint32_t>(fs.list[q][1])});
      }
      fb_fac_begin.push_back(static_cast<std::uint32_t>(fb_fac.size()));
    }
    buffer<Factor>(facs.size(), args.factors, facs.data());
    buffer<std::uint32_t>(fb_fac_begin.size(), args.fb_fac_begin, fb_fac_begin.data());
    buffer<uint_pair>(fb_fac.size(), args.fb_fac, fb_fac.data());

    // Parameters, written before each linearization.
    poses = mirrored<float>(8 * n_img, args.poses);
    rc = mirrored<float>(4 * n_img, args.rc);
    points = mirrored<float>(3 * npts, args.points);
    intr = mirrored<float>(np * n_blk, args.intr);
    basis = mirrored<float>(sys.basis.size(), args.basis);
    f_scale = mirrored<float>(sys.nf, args.f_scale);
    e_scale = mirrored<float>(sys.ne, args.e_scale);
    hres = mirrored<float>(2 * nobs, args.hres);
    res_cur.assign(2 * nobs, 0);
    res_cand.assign(2 * nobs, 0);

    // Linearization and step.
    buffer<float>(2 * nobs, args.r);
    buffer<float>(12 * nobs, args.Jp);
    buffer<float>(6 * nobs, args.Jx);
    buffer<float>(any_intr ? 2 * np * nobs : 1, args.Jk);
    const std::size_t nd = sys.has_depth ? nobs : 1;
    dr = mirrored<float>(nd, args.dr);
    dJp = mirrored<float>(6 * nd, args.dJp);
    dJx = mirrored<float>(3 * nd, args.dJx);
    dJc = mirrored<float>(2 * nd, args.dJc);
    fr = mirrored<float>(roff, args.fr);
    fJ = mirrored<float>(joff, args.fJ);
    buffer<float>(roff, args.fm);
    diag_f = mirrored<float>(sys.nf, args.diag_f);
    g_f = mirrored<float>(sys.nf, args.g_f);
    diag_e = mirrored<float>(sys.ne, args.diag_e);
    g_e = mirrored<float>(sys.ne, args.g_e);
    buffer<float>(sys.nf, args.d2_f);
    buffer<float>(sys.ne, args.d2_e);
    buffer<float>(9 * npts, args.Cinv);
    buffer<float>(sys.pinv_size, args.Pinv);
    buffer<float>(2 * nobs, args.t);
    buffer<float>(nd, args.td);
    buffer<float>(kChunkFloats * nchunks, args.partial);
    buffer<float>(2 * (npts + nfac), args.vals);
    bvec = mirrored<float>(sys.nf, args.b);
    x = mirrored<float>(sys.nf, args.x);
    buffer<float>(sys.nf, args.res);
    buffer<float>(sys.nf, args.z);
    buffer<float>(sys.nf, args.p);
    buffer<float>(sys.nf, args.q);
    dx_e = mirrored<float>(sys.ne, args.dx_e);

    if (direct) setup_direct();  // may leave strategy B
    if (!direct) {
      buffer<float>(1, args.Eo);
      buffer<float>(1, args.Go);
      buffer<Tile>(1, args.tiles);
      buffer<TileChunk>(1, args.tile_chunks);
      buffer<TileEntry>(1, args.entries);
      buffer<float>(1, args.tile_partial);
      tile_vals = buffer<float>(1, args.tile_vals);
    }

    args.nobs = static_cast<std::uint32_t>(nobs);
    args.npts = static_cast<std::uint32_t>(npts);
    args.nf = static_cast<std::uint32_t>(sys.nf);
    args.ne = static_cast<std::uint32_t>(sys.ne);
    args.nfb = static_cast<std::uint32_t>(sys.fblocks.size());
    args.nchunks = static_cast<std::uint32_t>(nchunks);
    args.ngunits = static_cast<std::uint32_t>(ngunits);
    args.ngsplits = static_cast<std::uint32_t>(ngsplits);
    args.nfac = static_cast<std::uint32_t>(nfac);
    args.np = static_cast<std::uint32_t>(np);
    args.ntiles = static_cast<std::uint32_t>(ntiles);
    args.ntile_chunks = static_cast<std::uint32_t>(ntile_chunks);
    args.has_depth = sys.has_depth ? 1 : 0;
    args.has_intr = any_intr ? 1 : 0;
    args.loss_type = static_cast<std::int32_t>(sys.loss.type);
    args.loss_scale = static_cast<float>(sys.loss.scale);
    args.pivot_floor = kPivotFloor;
    state_buf = dev->alloc(sizeof(State), nullptr, true);
    dev->bind(args, state_buf);
    linearize_k = linearize_kernel(sys.model);
  }

  // Strategy B's structure, counted before anything is built (setup_direct): the
  // blocks each observation touches by role, and per camera-side block A the tiles
  // (A, B), B >= A, with their entry counts.
  struct DirectPlan {
    std::vector<std::array<int, 3>> obs_blocks;  // per observation: pose, intrinsics, depth block (-1: none)
    std::vector<std::uint32_t> pos_img, pos_ib;  // each observation's place in img_list and ib_list
    // Per observation: whether it is the first of its point's observations in its
    // intrinsics block. That one's E and G (obs_schur_e, obs_schur_g) hold the sums over
    // all of them, since the block is shared.
    std::vector<char> intr_first;
    std::vector<std::vector<std::uint32_t>> fac_of;  // per block, the factors touching it, increasing
    std::vector<std::vector<std::pair<int, std::size_t>>> tiles_of;  // per block A: (B, entries), increasing B
    std::size_t ntiles = 0, entries = 0, tile_chunks = 0, nvals = 0, nnz = 0;
    std::size_t eo_floats = 0;  // E (and G) per observation and role
    std::size_t device_bytes = 0;
  };

  // The tile entries of block A, in the order the tiles' sums take them: for each
  // point that A's observations see, in increasing order, each observation ka of it
  // that touches A, each observation kb of it (only ka itself when the point is not
  // eliminated) and each role of kb, then the factors touching A. emit(B, entry) for
  // every B >= A. This is the order of the former map of lists, so the sums, and the
  // results, are unchanged where no intrinsics are free.
  //
  // An intrinsics role is shared by all of a point's observations in that block, so
  // its elimination terms factor: sum over a, b of G_a E_b^T is G_a times the sum of
  // the E_b, or the sum of the G_a times it. Those terms are taken once per point and
  // block, by the block's first observation of the point (intr_first), whose E and G
  // hold the sums. A pose-intrinsics entry then carries one observation's own J^T J
  // term and its G times the summed E; an intrinsics-intrinsics entry the own terms of
  // the whole run (bit 7) and the summed G times the summed E. Without this the pairs
  // of a point's observations would each add an entry to those tiles: 4.6 times the
  // entries on South Building with one shared block.
  template <typename Emit>
  void tile_entries(const DirectPlan& P, std::size_t A, Emit&& emit) const {
    std::size_t q0, q1;
    const std::vector<std::size_t>* list;
    sys.obs_range(sys.fblocks[A], q0, q1, list);
    const int a = static_cast<int>(A);
    std::size_t last = static_cast<std::size_t>(-1);
    for (std::size_t q = q0; q < q1; ++q) {
      const auto j = static_cast<std::size_t>(sys.o_pt[(*list)[q]]);
      if (j == last) continue;  // the list is in observation order, so by point
      last = j;
      const bool eliminated = sys.point_off[j] >= 0;
      for (std::size_t ka = sys.pt_begin[j]; ka < sys.pt_begin[j + 1]; ++ka) {
        const auto& ba = P.obs_blocks[ka];
        std::uint32_t ra = 3;
        for (std::uint32_t r = 0; r < 3; ++r)
          if (ba[r] == a) ra = r;
        if (ra == 3) continue;
        const std::uint32_t pa = ra == 1 ? P.pos_ib[ka] : P.pos_img[ka];
        const bool a_sum = ra != 1 || P.intr_first[ka];  // ka may stand for its point's intrinsics term
        for (std::size_t kb = sys.pt_begin[j]; kb < sys.pt_begin[j + 1]; ++kb) {
          const auto& bb = P.obs_blocks[kb];
          for (std::uint32_t rb = 0; rb < 3; ++rb) {
            if (bb[rb] < a) continue;  // also -1
            const std::uint32_t pb = rb == 1 ? P.pos_ib[kb] : P.pos_img[kb];
            if (rb == 1) {
              // kb's intrinsics block, once per point: by its first observation there.
              if (!P.intr_first[kb]) continue;
              if (ra == 1) {
                // Intrinsics with intrinsics: the run's own J^T J terms (one block: ka
                // and kb are then the same run), and the summed elimination term.
                if (!P.intr_first[ka]) continue;
                const bool own = ka == kb, elim = eliminated;
                if (!own && !elim) continue;
                emit(bb[rb], TileEntry{pa, pb, ra | (rb << 2) | (own ? 128u : 0u) | (elim ? 64u : 0u)});
              } else {
                // ka's role with the block: ka's own J^T J term when ka is in the block
                // (bit 5 reads both roles from ka), and G_a times the run's summed E.
                const bool own = ba[1] == bb[1], elim = eliminated;
                if (!own && !elim) continue;
                emit(bb[rb], TileEntry{pa, pb, ra | (rb << 2) | (own ? 32u : 0u) | (elim ? 64u : 0u)});
              }
              continue;
            }
            if (!eliminated && ka != kb) continue;
            const bool elim = eliminated && a_sum;
            if (ka != kb && !elim) continue;
            emit(bb[rb], TileEntry{pa, pb, ra | (rb << 2) | (ka == kb ? 32u : 0u) | (elim ? 64u : 0u)});
          }
        }
      }
    }
    for (const std::uint32_t f : P.fac_of[A]) {
      const auto& [fs, fl] = fac_index[f];
      for (int sa = 0; sa < fs->nslots; ++sa) {
        const int oa = fs->off(fl, sa);
        if (oa < 0 || block_at[static_cast<std::size_t>(oa)] != a) continue;
        for (int sb = 0; sb < fs->nslots; ++sb) {
          const int ob = fs->off(fl, sb);
          if (ob < 0) continue;
          const int B = block_at[static_cast<std::size_t>(ob)];
          if (B < a) continue;
          emit(B, TileEntry{f, 0, static_cast<std::uint32_t>(sa) | (static_cast<std::uint32_t>(sb) << 2) | 16u});
        }
      }
    }
  }

  DirectPlan plan_direct() {
    DirectPlan P;
    const std::size_t nfb = sys.fblocks.size(), nobs = sys.nobs;
    block_at.assign(sys.nf + 1, -1);  // f offset -> block
    for (std::size_t b = 0; b < nfb; ++b) block_at[static_cast<std::size_t>(sys.fblocks[b].offset)] = static_cast<int>(b);
    P.obs_blocks.resize(nobs);
    P.pos_img.resize(nobs);
    P.pos_ib.resize(nobs);
    sys.pool.parallel_for(nobs, [&](std::size_t k0, std::size_t k1) {
      for (std::size_t k = k0; k < k1; ++k) {
        auto& bl = P.obs_blocks[k];
        bl[0] = sys.o_pose[k] >= 0 ? block_at[static_cast<std::size_t>(sys.o_pose[k])] : -1;
        bl[1] = sys.o_intr[k] >= 0 ? block_at[static_cast<std::size_t>(sys.o_intr[k])] : -1;
        bl[2] = sys.has_depth && sys.o_dw[k] != 0 && sys.o_dep[k] >= 0
                    ? block_at[static_cast<std::size_t>(sys.o_dep[k])]
                    : -1;
        // img_list and ib_list are permutations: q -> k.
        P.pos_img[sys.img_list[k]] = static_cast<std::uint32_t>(k);
        P.pos_ib[sys.ib_list[k]] = static_cast<std::uint32_t>(k);
      }
    });
    P.intr_first.assign(nobs, 0);
    sys.pool.parallel_for(sys.pt_begin.size() - 1, [&](std::size_t j0, std::size_t j1) {
      for (std::size_t j = j0; j < j1; ++j)
        for (std::size_t k = sys.pt_begin[j]; k < sys.pt_begin[j + 1]; ++k) {
          const int c = P.obs_blocks[k][1];
          if (c < 0) continue;
          bool first = true;
          for (std::size_t q = sys.pt_begin[j]; q < k && first; ++q) first = P.obs_blocks[q][1] != c;
          P.intr_first[k] = first;
        }
    });
    fac_index.clear();
    P.fac_of.assign(nfb, {});
    for (const auto& fs : sys.fsets)
      for (std::size_t f = 0; f < fs.count; ++f) {
        const auto fg = static_cast<std::uint32_t>(fac_index.size());
        fac_index.push_back({&fs, f});
        for (int sl = 0; sl < fs.nslots; ++sl) {
          const int o = fs.off(f, sl);
          if (o < 0) continue;
          auto& l = P.fac_of[static_cast<std::size_t>(block_at[static_cast<std::size_t>(o)])];
          if (l.empty() || l.back() != fg) l.push_back(fg);
        }
      }

    // Count each block's tiles and entries, in parallel over the blocks.
    P.tiles_of.assign(nfb, {});
    sys.pool.parallel_for(
        nfb,
        [&](std::size_t a0, std::size_t a1) {
          std::vector<std::size_t> count(nfb, 0);
          std::vector<int> seen;
          for (std::size_t A = a0; A < a1; ++A) {
            seen.assign(1, static_cast<int>(A));  // every diagonal tile
            tile_entries(P, A, [&](int B, const TileEntry&) {
              if (count[static_cast<std::size_t>(B)]++ == 0 && B != static_cast<int>(A)) seen.push_back(B);
            });
            std::sort(seen.begin(), seen.end());
            auto& out = P.tiles_of[A];
            out.reserve(seen.size());
            for (const int B : seen) {
              out.push_back({B, count[static_cast<std::size_t>(B)]});
              count[static_cast<std::size_t>(B)] = 0;
            }
          }
        },
        1);

    for (std::size_t A = 0; A < nfb; ++A) {
      const auto& fa = sys.fblocks[A];
      for (const auto& [B, n] : P.tiles_of[A]) {
        const auto& fb = sys.fblocks[static_cast<std::size_t>(B)];
        ++P.ntiles;
        P.entries += n;
        P.tile_chunks += (n + kTileChunk - 1) / kTileChunk;
        P.nvals += static_cast<std::size_t>(fa.size * fb.size);
        for (int col = 0; col < fa.size; ++col)
          for (int y = 0; y < fb.size; ++y) P.nnz += fb.offset + y >= fa.offset + col;
      }
    }
    const std::size_t np = static_cast<std::size_t>(sys.np);
    P.eo_floats = 18 * nobs + (sys.has_depth ? 6 * nobs : 0) + (any_intr ? 3 * np * nobs : 0);
    P.device_bytes = P.ntiles * sizeof(Tile) + P.tile_chunks * (sizeof(TileChunk) + kChunkFloats * sizeof(float)) +
                     P.entries * sizeof(TileEntry) + 2 * P.eo_floats * sizeof(float) + P.nvals * sizeof(float);
    return P;
  }

  // Strategy B: the tile lists, the per-observation E and G, and the lower triangle in
  // compressed columns, analyzed once. Throws DoesNotFit where the structure passes a
  // 32-bit index or the device's free memory; with Auto it may instead leave
  // strategy A (direct = false).
  void setup_direct() {
    if (testing_fail_setup) {
      testing_fail_setup = false;
      throw DoesNotFit(dev->name() + ": sparse_schur set-up failed (test)");
    }
    DirectPlan P = plan_direct();
    const std::size_t nfb = sys.fblocks.size();
    std::string why;
    if (P.entries > 0xffffffffu || P.nvals > 0xffffffffu || P.eo_floats > 0xffffffffu ||
        P.nnz > 0x7fffffffu || kChunkFloats * P.tile_chunks > 0xffffffffu)
      why = "the reduced camera system passes sparse_schur's 32-bit indices (" + std::to_string(P.entries) +
            " tile entries, " + std::to_string(P.nnz) + " non-zeros)";
    else if (P.device_bytes > dev->free_memory() / 10 * 8)
      why = "the reduced camera system needs " + std::to_string(P.device_bytes >> 20) + " MiB of device memory, " +
            std::to_string(dev->free_memory() >> 20) + " MiB are free";
    if (!why.empty() && !allow_fallback) throw DoesNotFit(dev->name() + ": " + why);
    if (!why.empty()) {
      direct = false;
      solver.reset();
      note = "auto: iterative_schur, as " + why;
      return;
    }

    // Tiles in order of column block, then row block, and their entries.
    std::vector<Tile> tiles(P.ntiles);
    std::vector<TileChunk> tchunks;
    tchunks.reserve(P.tile_chunks);
    std::vector<std::size_t> tile_first(nfb + 1, 0), entry_first(P.ntiles + 1, 0);
    {
      std::size_t t = 0, nvals = 0;
      for (std::size_t A = 0; A < nfb; ++A) {
        tile_first[A] = t;
        for (const auto& [B, n] : P.tiles_of[A]) {
          Tile& T = tiles[t];
          T.a = static_cast<std::uint32_t>(A);
          T.b = static_cast<std::uint32_t>(B);
          T.c0 = static_cast<std::uint32_t>(tchunks.size());
          const std::size_t e0 = entry_first[t];
          for (std::size_t e = e0; e < e0 + n; e += kTileChunk)
            tchunks.push_back({static_cast<std::uint32_t>(t), static_cast<std::uint32_t>(e),
                               static_cast<std::uint32_t>(std::min(e0 + n, e + kTileChunk))});
          T.c1 = static_cast<std::uint32_t>(tchunks.size());
          T.voff = static_cast<std::uint32_t>(nvals);
          nvals += static_cast<std::size_t>(sys.fblocks[A].size * sys.fblocks[static_cast<std::size_t>(B)].size);
          entry_first[t + 1] = e0 + n;
          ++t;
        }
      }
      tile_first[nfb] = t;
    }
    {
      std::vector<TileEntry> entries(P.entries);
      sys.pool.parallel_for(
          nfb,
          [&](std::size_t a0, std::size_t a1) {
            std::vector<std::size_t> cursor(nfb, 0);
            for (std::size_t A = a0; A < a1; ++A) {
              for (std::size_t t = tile_first[A]; t < tile_first[A + 1]; ++t) cursor[tiles[t].b] = entry_first[t];
              tile_entries(P, A, [&](int B, const TileEntry& e) { entries[cursor[static_cast<std::size_t>(B)]++] = e; });
            }
          },
          1);
      buffer<TileEntry>(entries.size(), args.entries, entries.data());
    }
    ntiles = tiles.size();
    ntile_chunks = tchunks.size();
    buffer<Tile>(tiles.size(), args.tiles, tiles.data());
    buffer<TileChunk>(tchunks.size(), args.tile_chunks, tchunks.data());
    buffer<float>(kChunkFloats * tchunks.size(), args.tile_partial);
    tile_vals = mirrored<float>(P.nvals, args.tile_vals);
    // Each observation's E and G per role (obs_schur_e, obs_schur_g), by list place.
    const std::size_t n_dep = sys.has_depth ? 6 * sys.nobs : 0;
    args.eo_dep = static_cast<std::uint32_t>(18 * sys.nobs);
    args.eo_intr = static_cast<std::uint32_t>(18 * sys.nobs + n_dep);
    buffer<float>(P.eo_floats, args.Eo);
    buffer<float>(P.eo_floats, args.Go);

    // Lower triangle, compressed columns: column c of block A holds rows r >= c of
    // every tile (A, B), B >= A, in increasing B. Columns follow the blocks' order.
    std::vector<std::size_t> nnz_first(nfb + 1, 0);
    for (std::size_t A = 0; A < nfb; ++A) {
      const auto& fa = sys.fblocks[A];
      std::size_t n = 0;
      for (const auto& [B, cnt] : P.tiles_of[A]) {
        const auto& fb = sys.fblocks[static_cast<std::size_t>(B)];
        for (int col = 0; col < fa.size; ++col)
          for (int y = 0; y < fb.size; ++y) n += fb.offset + y >= fa.offset + col;
      }
      nnz_first[A + 1] = nnz_first[A] + n;
    }
    csc_begin.assign(sys.nf + 1, 0);
    csc_rows.resize(P.nnz);
    csc_src.resize(P.nnz);
    sys.pool.parallel_for(
        nfb,
        [&](std::size_t a0, std::size_t a1) {
          for (std::size_t A = a0; A < a1; ++A) {
            const auto& fa = sys.fblocks[A];
            std::size_t at = nnz_first[A];
            for (int col = 0; col < fa.size; ++col) {
              const std::size_t c = static_cast<std::size_t>(fa.offset + col);
              for (std::size_t t = tile_first[A]; t < tile_first[A + 1]; ++t) {
                const Tile& T = tiles[t];
                const auto& fb = sys.fblocks[T.b];
                for (int y = 0; y < fb.size; ++y) {
                  const int r = fb.offset + y;
                  if (r < static_cast<int>(c)) continue;
                  csc_rows[at] = r;
                  csc_src[at] = T.voff + static_cast<std::uint32_t>(col * fb.size + y);
                  ++at;
                }
              }
              csc_begin[c + 1] = static_cast<std::int64_t>(at);
            }
          }
        },
        1);
    csc_vals.assign(csc_rows.size(), 0.0);
    rhs.assign(sys.nf, 0.0);
    if (sys.nf == 0) return;
    solver->analyze(sys.nf, csc_begin, csc_rows);
  }

  // Factorizes the tiles the device formed and solves S dx_f = b into x.
  FactorStatus direct_solve() {
    // Nothing free on the camera side (only points are refined): the reduced system
    // is empty, dx_f has no entries, and the solver must not see the empty structure.
    if (sys.nf == 0) return FactorStatus::Solved;
    dev->to_host(tile_vals);
    dev->to_host(bvec);
    const float* tv = host(tile_vals);
    sys.pool.parallel_for(csc_vals.size(), [&](std::size_t i0, std::size_t i1) {
      for (std::size_t i = i0; i < i1; ++i) csc_vals[i] = static_cast<double>(tv[csc_src[i]]);
    });
    const float* bf = host(bvec);
    for (std::size_t c = 0; c < sys.nf; ++c) rhs[c] = static_cast<double>(bf[c]);
    FactorStatus st = solver->factor_solve(csc_vals, rhs);
    if (testing_factor_failures > 0) {
      --testing_factor_failures;
      st = testing_factor_status;
    }
    if (st != FactorStatus::Solved) return st;
    float* xf = host(x);
    for (std::size_t c = 0; c < sys.nf; ++c) xf[c] = static_cast<float>(rhs[c]);
    dev->to_device(x);
    return FactorStatus::Solved;
  }

  // ---------------------------------------------------------------------------
  // Dispatch helpers.

  void run(Kernel k, std::size_t n, const Op& op = {}) {
    if (n > 0) dev->dispatch_threads(k, n, op);
  }

  // One group: a deterministic reduction into a state slot.
  void reduce(int mode, const float* a, const float* b, std::size_t n, int slot, int iter) {
    Op op{};
    op.x = a;
    op.y = const_cast<float*>(b);
    op.n = static_cast<std::uint32_t>(n);
    op.mode = mode;
    op.slot = slot;
    op.iter = iter;
    dev->dispatch_groups(Kernel::Reduce, 1, kReduceThreads, op);
  }

  // One group per gather unit, then the split blocks.
  void gather(int mode, const float* vx = nullptr, float* vy = nullptr, int iter = 0) {
    if (ngunits == 0) return;
    Op op{};
    op.x = vx;
    op.y = vy;
    op.mode = mode;
    op.iter = iter;
    dev->dispatch_groups(Kernel::BlockGather, ngunits, kGatherThreads, op);
    run(Kernel::GatherFinish, ngsplits, op);
  }

  // One group per chunk of a tile's entries, then each tile's values.
  void tiles() {
    run(Kernel::ObsSchurG, any_intr ? 2 * sys.nobs : sys.nobs);
    if (ntiles == 0) return;
    if (ntile_chunks > 0) dev->dispatch_groups(Kernel::SchurTiles, ntile_chunks, kTileThreads, Op{});
    run(Kernel::TileFinish, kChunkFloats * ntiles);
  }

  // Squared column norms and the gradient, both sides.
  void encode_colnorm() {
    gather(0);
    run(Kernel::PointColnorm, npts);
  }

  // y = S x for camera-side vectors x, y (inside the conjugate gradients loop).
  void encode_schur(const float* vx, float* vy, int iter) {
    Op op{};
    op.x = vx;
    op.y = vy;
    op.iter = iter;
    op.mode = 1;
    run(Kernel::SchurPoints, npts + nfac, op);  // the factors' J x too
    gather(1, vx, vy, iter);
  }

  void encode_pcg_iteration(int i) {
    Op it{};
    it.iter = i;
    run(Kernel::PcgPrecondition, sys.fblocks.size(), it);
    reduce(1, args.res, args.z, sys.nf, jba_gpu::kSlotRho, i);
    run(Kernel::PcgRho, 1, it);
    run(Kernel::PcgDirection, sys.nf, it);
    encode_schur(args.p, args.q, i);
    reduce(1, args.p, args.q, sys.nf, jba_gpu::kSlotPQ, i);
    run(Kernel::PcgAlpha, 1, it);
    const bool reset = i % kResidualResetPeriod == 0;
    Op up = it;
    up.mode = reset ? 1 : 0;
    run(Kernel::PcgUpdate, sys.nf, up);
    if (reset) {
      encode_schur(args.x, args.q, i);
      run(Kernel::PcgReset, sys.nf, it);
    }
    reduce(2, args.x, nullptr, sys.nf, jba_gpu::kSlotQ, i);
    run(Kernel::PcgTest, 1, it);
  }

  // ---------------------------------------------------------------------------

  void upload_parameters() {
    const auto& P = sys.cur;
    const std::size_t n_img = sys.pose_off.size();
    float* po = host(poses);
    float* rcp = host(rc);
    for (std::size_t i = 0; i < n_img; ++i) {
      const double* pose = &P.poses[7 * i];
      double Rc[3];
      quat_rotate(pose, centre, Rc);
      for (int c = 0; c < 4; ++c) po[8 * i + c] = static_cast<float>(pose[c]);
      for (int c = 0; c < 3; ++c) {
        po[8 * i + 4 + c] = static_cast<float>(pose[4 + c] + Rc[c]);
        rcp[4 * i + c] = static_cast<float>(Rc[c]);
      }
    }
    float* pts = host(points);
    sys.pool.parallel_for(npts, [&](std::size_t j0, std::size_t j1) {
      for (std::size_t j = j0; j < j1; ++j)
        for (int c = 0; c < 3; ++c) pts[3 * j + c] = static_cast<float>(P.points[3 * j + c] - centre[c]);
    });
    float* k = host(intr);
    for (std::size_t c = 0; c < P.intr.size(); ++c) k[c] = static_cast<float>(P.intr[c]);
    float* B = host(basis);
    for (std::size_t c = 0; c < sys.basis.size(); ++c) B[c] = static_cast<float>(sys.basis[c]);
    for (const Buffer* b : {&poses, &rc, &points, &intr, &basis}) dev->to_device(*b);
  }

  // Depth rows and factors in double on the host, unscaled.
  void host_rows() {
    if (sys.has_depth) {
      float *r = host(dr), *jp = host(dJp), *jx = host(dJx), *jc = host(dJc);
      sys.pool.parallel_for(sys.nobs, [&](std::size_t k0, std::size_t k1) {
        for (std::size_t k = k0; k < k1; ++k) {
          if (sys.o_dw[k] == 0) continue;
          const auto i = static_cast<std::size_t>(sys.o_img[k]);
          const std::uint8_t pm = sys.o_pose[k] < 0 ? std::uint8_t(0x3f) : sys.pose_fixed_cols[i];
          double rd, jpd[6], jxd[3], jcd[2];
          sys.linearize_depth_row(k, pm, &rd, jpd, jxd, jcd);
          r[k] = static_cast<float>(rd);
          for (int c = 0; c < 6; ++c) jp[6 * k + c] = static_cast<float>(jpd[c]);
          for (int c = 0; c < 3; ++c) jx[3 * k + c] = static_cast<float>(jxd[c]);
          for (int c = 0; c < 2; ++c) jc[2 * k + c] = static_cast<float>(jcd[c]);
        }
      });
      for (const Buffer* b : {&dr, &dJp, &dJx, &dJc}) dev->to_device(*b);
    }
    if (nfac == 0) return;
    sys.linearize_factors();
    float *r = host(fr), *J = host(fJ);
    for (std::size_t s = 0; s < sys.fsets.size(); ++s) {
      const auto& fs = sys.fsets[s];
      for (std::size_t a = 0; a < fs.r.size(); ++a) r[set_roff[s] + a] = static_cast<float>(fs.r[a]);
      for (std::size_t a = 0; a < fs.J.size(); ++a) J[set_joff[s] + a] = static_cast<float>(fs.J[a]);
    }
    dev->to_device(fr);
    dev->to_device(fJ);
  }

  // Cost in double at P (System::cost_of), keeping the whitened reprojection
  // residuals in `out`.
  double cost_of(const cpu::Params& P, std::vector<float>& out) {
    const std::size_t np = static_cast<std::size_t>(sys.np);
    double c = 0.5 * sys.pool.sum(sys.nobs, [&](std::size_t k) {
      const auto i = static_cast<std::size_t>(sys.o_img[k]);
      const double* pose = &P.poses[7 * i];
      double res[2];
      cpu::reprojection<double>(sys.model, pose, pose + 4, &P.points[3 * static_cast<std::size_t>(sys.o_pt[k])],
                                &P.intr[np * static_cast<std::size_t>(sys.image_block[i])], &sys.o_xy[2 * k], res,
                                nullptr, nullptr, nullptr);
      const double w = sys.o_w[k];
      out[2 * k] = static_cast<float>(res[0] * w);
      out[2 * k + 1] = static_cast<float>(res[1] * w);
      const double s = (res[0] * res[0] + res[1] * res[1]) * w * w;
      double rho[3];
      evaluate_loss(sys.loss, s, rho);
      return rho[0];
    });
    if (sys.has_depth) c += sys.depth_cost(P);
    for (const auto& fs : sys.fsets) c += sys.factor_cost(fs, P);
    return c;
  }

  void linearize(bool compute_scaling) {
    sys.update_bases();
    if (!res_cur_valid) cost_of(sys.cur, res_cur);
    res_cur_valid = true;
    std::memcpy(hres.host, res_cur.data(), res_cur.size() * sizeof(float));
    dev->to_device(hres);
    upload_parameters();
    host_rows();
    dev->begin();
    run(linearize_k, sys.nobs);
    if (compute_scaling) encode_colnorm();
    dev->finish();
    if (compute_scaling) {
      // Ceres: scale = 1 / (1 + column norm), from the first Jacobian.
      dev->to_host(diag_f);
      dev->to_host(diag_e);
      const float* df = host(diag_f);
      const float* de = host(diag_e);
      for (std::size_t c = 0; c < sys.nf; ++c) sys.f_scale[c] = 1 / (1 + std::sqrt(static_cast<double>(df[c])));
      for (std::size_t c = 0; c < sys.ne; ++c) sys.e_scale[c] = 1 / (1 + std::sqrt(static_cast<double>(de[c])));
      float* fs = host(f_scale);
      float* es = host(e_scale);
      for (std::size_t c = 0; c < sys.nf; ++c) fs[c] = static_cast<float>(sys.f_scale[c]);
      for (std::size_t c = 0; c < sys.ne; ++c) es[c] = static_cast<float>(sys.e_scale[c]);
      dev->to_device(f_scale);
      dev->to_device(e_scale);
      sys.have_scaling = true;
    }
    dev->begin();
    if (sys.have_scaling) {
      run(Kernel::ScaleObservations, sys.nobs);
      run(Kernel::ScaleFactors, nfac);
    }
    encode_colnorm();
    if (direct) run(Kernel::ObsSchurE, any_intr ? 2 * sys.nobs : sys.nobs);
    dev->finish();
    dev->to_host(g_f);
    dev->to_host(g_e);
    const float* gf = host(g_f);
    const float* ge = host(g_e);
    double m = 0;
    for (std::size_t c = 0; c < sys.nf; ++c) m = std::max(m, std::abs(gf[c] / sys.f_scale[c]));
    for (std::size_t c = 0; c < sys.ne; ++c) m = std::max(m, std::abs(ge[c] / sys.e_scale[c]));
    gmax = m;
  }

  StepResult compute_step(double radius, double eta, int min_it, int max_it) {
    StepResult out;
    State& s = state();
    std::memset(&s, 0, sizeof(State));
    s.eta = static_cast<float>(eta);
    s.min_it = min_it;
    s.max_it = max_it;
    dev->to_device(state_buf);
    {
      dev->begin();
      Op damp{};
      damp.radius = static_cast<float>(radius);
      run(Kernel::Damping, sys.nf + sys.ne, damp);
      run(Kernel::PointBlocks, npts);
      run(Kernel::RhsPoints, npts);
      gather(3);
      if (direct) tiles();
      dev->finish();
    }
    bool iterative = !direct;
    if (direct) {
      // Strategy B: factorize the reduced camera system in double. Formed in single
      // precision, it can lose definiteness where points have a degenerate depth
      // direction; that step then falls back to conjugate gradients on the implicit
      // system, which tolerates it. A factorization that ran out of memory ends
      // strategy B for the solve (Auto) or the solve (sparse_schur asked for).
      out.linear_iterations = 1;
      if (s.bad) return out;
      const FactorStatus st = direct_solve();
      if (st == FactorStatus::OutOfMemory) {
        if (!allow_fallback)
          throw DoesNotFit(dev->name() + ": sparse_schur ran out of device memory in the factorization; "
                                         "iterative_schur needs less");
        direct = false;
        solver.reset();
        note = "auto: sparse_schur ran out of device memory in the factorization; iterative_schur from there on";
      }
      // A failure that is not about the matrix (FactorStatus::Failed) falls back
      // too, but three in a row mean the solver itself is broken (document 15,
      // finding 4).
      failed_in_a_row = st == FactorStatus::Failed ? failed_in_a_row + 1 : 0;
      if (failed_in_a_row >= kMaxFailedFactorizations)
        throw std::runtime_error(dev->name() + ": sparse_schur's factorization failed " +
                                 std::to_string(failed_in_a_row) + " times in a row (not a matrix property)");
      if (st != FactorStatus::Solved) {
        iterative = true;
        ++fallback_steps;
      }
    }
    if (iterative) {
      {
        dev->begin();
        Op pre{};
        pre.mode = 2;
        run(Kernel::ChunkPartial, nchunks, pre);
        run(Kernel::BlockFinish, sys.fblocks.size(), pre);
        run(Kernel::PcgVectorsInit, sys.nf);
        reduce(1, args.b, args.b, sys.nf, jba_gpu::kSlotBB, 0);
        run(Kernel::PcgInit, 1);
        dev->finish();
      }
      // Conjugate gradients in batches of iterations; after the loop has stopped,
      // the rest of a batch does nothing.
      int i = 1, batch = 4;
      while (!s.done && i <= max_it) {
        dev->begin();
        for (int n = 0; n < batch && i <= max_it; ++n, ++i) encode_pcg_iteration(i);
        dev->finish();
        batch = std::min(batch * 2, 64);
      }
      out.linear_iterations = s.iterations;
    }
    if (s.failed) return out;
    {
      dev->begin();
      run(Kernel::BackSubstitute, npts);
      Op fx{};
      fx.x = args.x;
      run(Kernel::FactorsTimes, nfac, fx);
      run(Kernel::FactorModel, nfac);
      reduce(0, args.vals, nullptr, npts + nfac, jba_gpu::kSlotModel, 0);
      // vals: the terms (npts + nfac), then their magnitudes (back_substitute, factor_model).
      reduce(0, args.vals + (npts + nfac), nullptr, npts + nfac, jba_gpu::kSlotModelAbs, 0);
      dev->finish();
    }
    out.model_cost_change = -static_cast<double>(s.slot[jba_gpu::kSlotModel]);
    const double resolvable = kResolvable * static_cast<double>(s.slot[jba_gpu::kSlotModelAbs]);
    dev->to_host(x);
    dev->to_host(dx_e);
    out.step_norm = sys.retract(host(x), host(dx_e));
    out.valid = std::isfinite(out.model_cost_change) && out.model_cost_change > 0 && std::isfinite(out.step_norm);
    if (std::isfinite(out.model_cost_change) && std::abs(out.model_cost_change) <= resolvable) {
      out.valid = false;
      out.precision_limit = true;
    }
    return out;
  }
};

GpuBackend::GpuBackend(const Problem& problem, const SolverOptions& options, const Gauge& gauge,
                       const std::function<std::unique_ptr<Device>()>& make_device) {
  try {
    impl_ = std::make_unique<Impl>(problem, options, gauge, make_device(), true);
  } catch (const DoesNotFit& e) {
    // Auto's strategy B did not fit after all (an allocation, or the factorization's
    // analysis): again on a fresh device, which frees the first one's memory.
    if (options.linear_solver != LinearSolverType::Auto) throw;
    const std::string why = e.what();
    impl_.reset();
    impl_ = std::make_unique<Impl>(problem, options, gauge, make_device(), false);
    impl_->note = "auto: iterative_schur, as sparse_schur did not fit (" + why + ")";
  }
}

LinearSolverType GpuBackend::linear_solver() const {
  return impl_->direct ? LinearSolverType::SparseSchur : LinearSolverType::IterativeSchur;
}

std::string GpuBackend::note() const { return impl_->note; }

int GpuBackend::fallback_steps() const { return impl_->fallback_steps; }

namespace testing {
void fail_factorizations(int count, FactorStatus status) {
  testing_factor_failures = count;
  testing_factor_status = status;
}
void fail_direct_setup(bool fail) { testing_fail_setup = fail; }
}  // namespace testing

GpuBackend::~GpuBackend() = default;

double GpuBackend::cost() {
  impl_->res_cur_valid = true;
  return impl_->cost_of(impl_->sys.cur, impl_->res_cur);
}

void GpuBackend::cost_by_type(double* out) { impl_->sys.cost_by_type(out); }

void GpuBackend::linearize(bool compute_scaling) {
  impl_->dev->scoped([&] { impl_->linearize(compute_scaling); });
}

double GpuBackend::gradient_max_norm() const { return impl_->gmax; }

double GpuBackend::parameter_norm() const { return impl_->sys.parameter_norm(); }

StepResult GpuBackend::compute_step(double radius, double eta, int min_linear_iterations, int max_linear_iterations) {
  StepResult out;
  impl_->dev->scoped([&] { out = impl_->compute_step(radius, eta, min_linear_iterations, max_linear_iterations); });
  return out;
}

bool GpuBackend::candidate_cost(double* cost) {
  *cost = impl_->cost_of(impl_->sys.cand, impl_->res_cand);
  return std::isfinite(*cost);
}

void GpuBackend::accept_candidate() {
  impl_->sys.cur.swap(impl_->sys.cand);
  impl_->res_cur.swap(impl_->res_cand);
}

void GpuBackend::write_back(Problem& problem) const { impl_->sys.write_back(problem); }

}  // namespace jba::gpu
