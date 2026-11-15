#include "jba/metal/backend.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "jba/cpu/reprojection.h"
#include "jba/cpu/system.h"
#include "jba/loss.h"
#include "jba/geometry.h"

// Embedded at build time from residuals/generated/metal/reprojection.metal.h and
// backends/metal/kernels/solver.metal (see backends/metal/CMakeLists.txt).
extern const char* const kJbaMetalSolverSource;

namespace jba::metal {
namespace {

// Host mirrors of the kernels' structures (solver.metal); checked against the
// kernels' own sizes when the library is compiled.
using Ptr = std::uint64_t;  // a buffer's GPU address

struct FBlockDesc {
  std::int32_t offset, size, pinv_offset, kind;
};

struct Chunk {
  std::uint32_t block, list, q0, q1;
};

struct Factor {
  std::int32_t rows, cols, joff, roff, nslots;
  std::int32_t off[3], base[3], size[3];
};

struct Tile {
  std::uint32_t a, b, c0, c1, voff;
};

struct TileChunk {
  std::uint32_t tile, e0, e1;
};

struct TileEntry {
  std::uint32_t ka, kb, info;
};

struct Args {
  Ptr o_img, o_pt, o_xy, o_w, o_pose, o_intr, o_dw, o_dep;
  Ptr pt_begin, point_off, image_block, pose_off, intr_off, pose_mask, intr_mask, basis_of;
  Ptr img_list, ib_list, chunks, block_chunks, fblocks, factors, fb_fac_begin, fb_fac;
  Ptr poses, rc, points, intr, basis, f_scale, e_scale, hres;
  Ptr r, Jp, Jx, Jk, dr, dJp, dJx, dJc, fr, fJ, fm, diag_f, g_f, diag_e, g_e;
  Ptr d2_f, d2_e, Cinv, Pinv, t, td, partial, vals, b, x, res, z, p, q, dx_e;
  Ptr tiles, tile_chunks, entries, tile_partial, tile_vals;
  std::uint32_t nobs, npts, nf, ne, nfb, nchunks, nfac, np, ntiles, ntile_chunks;
  std::int32_t has_depth, has_intr, loss_type;
  float loss_scale, pivot_floor;
};

struct State {
  float slot[8];
  float rho, rho_prev, pq, alpha, beta, Q0, eta, model_change;
  std::int32_t min_it, max_it, done, failed, iterations, pad0, pad1, pad2;
  std::int32_t bad;
};

struct Op {
  Ptr x = 0, y = 0;
  std::uint32_t n = 0;
  std::int32_t mode = 0, slot = 0, iter = 0;
  float radius = 0, pad = 0;
};

enum Slot { kSlotBB = 0, kSlotRho = 1, kSlotPQ = 2, kSlotQ = 3, kSlotModel = 4, kSlotModelAbs = 5 };

// A predicted decrease smaller than this share of the sum of its terms' magnitudes
// cannot be resolved by the single-precision sums that form it.
constexpr double kResolvable = 1e-5;

// Cholesky pivots of the point and camera blocks below this share of their diagonal
// entry are raised to it (solver.metal, spd_inverse): about eight single-precision
// epsilons, the rounding error of a pivot itself, so a pivot below it carries no
// information. Measured on BAL ladybug-49: 1e-7 lets points with a degenerate depth
// direction take steps of 1e5; 1e-5 and above slow convergence (damping where the
// block is in fact resolved).
constexpr float kPivotFloor = 1e-6f;

constexpr std::size_t kChunkFloats = 64;
constexpr std::size_t kChunkObservations = 32;
constexpr NSUInteger kReduceThreads = 256;
constexpr NSUInteger kGatherThreads = 256;  // solver.metal, block_gather
constexpr NSUInteger kTileThreads = 64;     // solver.metal, schur_tiles
constexpr std::size_t kTileChunk = 1024;   // entries per threadgroup of schur_tiles
constexpr int kResidualResetPeriod = 10;  // Ceres's conjugate gradients default

[[noreturn]] void fail(const std::string& what, NSError* error) {
  std::string msg = "metal: " + what;
  if (error != nil) msg += ": " + std::string(error.localizedDescription.UTF8String);
  throw std::runtime_error(msg);
}

const char* model_suffix(CameraModel model) {
  switch (model) {
    case CameraModel::SimplePinhole: return "simple_pinhole";
    case CameraModel::Pinhole: return "pinhole";
    case CameraModel::SimpleRadial: return "simple_radial";
    case CameraModel::Radial: return "radial";
    case CameraModel::OpenCV: return "opencv";
    case CameraModel::OpenCVFisheye: return "opencv_fisheye";
  }
  return "pinhole";
}

constexpr const char* kKernels[] = {
    "linearize_simple_pinhole", "linearize_pinhole", "linearize_simple_radial", "linearize_radial",
    "linearize_opencv", "linearize_opencv_fisheye", "scale_observations", "scale_factors", "chunk_partial",
    "block_finish", "block_gather", "point_colnorm", "damping", "point_blocks", "schur_points", "factors_times", "rhs_points",
    "back_substitute", "factor_model", "reduce", "pcg_vectors_init", "pcg_init", "pcg_precondition", "pcg_rho",
    "pcg_direction", "pcg_alpha", "pcg_update", "pcg_reset", "pcg_test", "schur_tiles", "tile_finish", "layout_sizes"};

// The compiled kernels, once per process (D50: compiled at run time from embedded
// source; pipelines are cached so a second solve pays nothing).
struct Library {
  id<MTLDevice> device = nil;
  std::map<std::string, id<MTLComputePipelineState>> pipelines;
};

const Library& library() {
  static Library lib;
  static std::once_flag once;
  static std::string error;
  std::call_once(once, [] {
    @autoreleasepool {
      try {
        lib.device = MTLCreateSystemDefaultDevice();
        if (lib.device == nil) fail("no Metal device", nil);
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion3_0;  // device pointers in argument structures
        // No fast math: the generated arithmetic must match the CPU emission.
        if (@available(macOS 15.0, *)) {
          options.mathMode = MTLMathModeSafe;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
          options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        NSError* err = nil;
        id<MTLLibrary> mtl = [lib.device newLibraryWithSource:@(kJbaMetalSolverSource) options:options error:&err];
        if (mtl == nil) fail("kernel compilation failed", err);
        for (const char* name : kKernels) {
          id<MTLFunction> fn = [mtl newFunctionWithName:@(name)];
          if (fn == nil) fail(std::string("kernel not found: ") + name, nil);
          id<MTLComputePipelineState> pso = [lib.device newComputePipelineStateWithFunction:fn error:&err];
          if (pso == nil) fail(std::string("pipeline creation failed: ") + name, err);
          lib.pipelines[name] = pso;
        }
        if (lib.pipelines["reduce"].maxTotalThreadsPerThreadgroup < kReduceThreads)
          fail("the reduction needs 256 threads per threadgroup", nil);
        // The host's structure layouts must match the kernels'.
        id<MTLBuffer> sizes = [lib.device newBufferWithLength:64 options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> queue = [lib.device newCommandQueue];
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:lib.pipelines["layout_sizes"]];
        [enc setBuffer:sizes offset:0 atIndex:0];
        [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const auto* s = static_cast<const std::uint32_t*>(sizes.contents);
        const std::uint32_t want[] = {sizeof(Args), sizeof(State), sizeof(Op), sizeof(Factor), sizeof(Chunk),
                                      sizeof(FBlockDesc), sizeof(Tile), sizeof(TileEntry), sizeof(TileChunk)};
        for (int i = 0; i < 9; ++i)
          if (s[i] != want[i])
            fail("structure " + std::to_string(i) + " is " + std::to_string(s[i]) + " bytes in the kernels, " +
                     std::to_string(want[i]) + " on the host",
                 nil);
      } catch (const std::exception& e) {
        error = e.what();
        lib.device = nil;
      }
    }
  });
  if (lib.device == nil) throw std::runtime_error(error.empty() ? "metal: no Metal device" : error);
  return lib;
}

}  // namespace

struct MetalBackend::Impl {
  cpu::System sys;
  const Library& lib;
  id<MTLCommandQueue> queue;
  std::vector<id<MTLResource>> resources;
  id<MTLBuffer> args_buf, state_buf;
  id<MTLComputePipelineState> linearize_pso;

  // Buffers the host writes or reads.
  id<MTLBuffer> poses, rc, points, intr, basis, f_scale, e_scale, hres;
  id<MTLBuffer> dr, dJp, dJx, dJc, fr, fJ;
  id<MTLBuffer> diag_f, g_f, diag_e, g_e, x, dx_e, bvec;
  Args args{};

  double centre[3] = {0, 0, 0};
  bool any_intr = false;
  std::size_t nchunks = 0, nfac = 0, npts = 0;
  std::vector<std::size_t> set_joff, set_roff;  // per factor set, into fJ and fr
  double gmax = 0;
  // Whitened reprojection residuals in double rounded to float, at the current and
  // the candidate parameters: by-products of their costs.
  std::vector<float> res_cur, res_cand;
  bool res_cur_valid = false;

  // Strategy B: the reduced camera system's tiles on the GPU, its lower triangle in
  // compressed columns for Accelerate (csc_src: each value's place among the tiles'),
  // and the symbolic factorization, computed once.
  bool direct = false;
  std::size_t ntiles = 0, ntile_chunks = 0;
  id<MTLBuffer> tile_vals;
  std::vector<long> csc_begin;
  std::vector<int> csc_rows;
  std::vector<std::uint32_t> csc_src;
  std::vector<double> csc_vals, rhs;
  SparseOpaqueSymbolicFactorization symbolic{};
  bool have_symbolic = false;

  ~Impl() {
    if (have_symbolic) SparseCleanup(symbolic);
  }

  Impl(const Problem& p, const SolverOptions& o, const Gauge& gauge)
      : sys(p, o, gauge), lib(library()), direct(o.linear_solver == LinearSolverType::SparseSchur) {
    @autoreleasepool {
      queue = [lib.device newCommandQueue];
      if (queue == nil) fail("no command queue", nil);
      setup(p);
    }
  }

  template <typename T>
  id<MTLBuffer> buffer(std::size_t count, Ptr& address, const T* init = nullptr) {
    const std::size_t bytes = std::max<std::size_t>(count * sizeof(T), 16);
    id<MTLBuffer> b = [lib.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (b == nil) fail("buffer allocation failed (" + std::to_string(bytes) + " bytes)", nil);
    if (init != nullptr && count > 0) std::memcpy(b.contents, init, count * sizeof(T));
    else std::memset(b.contents, 0, bytes);
    address = b.gpuAddress;
    resources.push_back(b);
    return b;
  }

  template <typename T, typename S>
  id<MTLBuffer> converted(const std::vector<S>& v, Ptr& address) {
    std::vector<T> c(v.begin(), v.end());
    return buffer<T>(c.size(), address, c.data());
  }

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
    std::vector<std::uint32_t> fb_fac;  // pairs
    for (std::size_t b = 0; b < sys.fblocks.size(); ++b) {
      for (std::size_t s = 0; s < sys.fsets.size(); ++s) {
        const auto& fs = sys.fsets[s];
        for (std::size_t q = fs.begin[b]; q < fs.begin[b + 1]; ++q) {
          fb_fac.push_back(static_cast<std::uint32_t>(set_first[s] + static_cast<std::size_t>(fs.list[q][0])));
          fb_fac.push_back(static_cast<std::uint32_t>(fs.list[q][1]));
        }
      }
      fb_fac_begin.push_back(static_cast<std::uint32_t>(fb_fac.size() / 2));
    }
    buffer<Factor>(facs.size(), args.factors, facs.data());
    buffer<std::uint32_t>(fb_fac_begin.size(), args.fb_fac_begin, fb_fac_begin.data());
    buffer<std::uint32_t>(fb_fac.size(), args.fb_fac, fb_fac.data());

    // Parameters, written before each linearization.
    poses = buffer<float>(8 * n_img, args.poses);
    rc = buffer<float>(4 * n_img, args.rc);
    points = buffer<float>(3 * npts, args.points);
    intr = buffer<float>(np * n_blk, args.intr);
    basis = buffer<float>(sys.basis.size(), args.basis);
    f_scale = buffer<float>(sys.nf, args.f_scale);
    e_scale = buffer<float>(sys.ne, args.e_scale);
    hres = buffer<float>(2 * nobs, args.hres);
    res_cur.assign(2 * nobs, 0);
    res_cand.assign(2 * nobs, 0);

    // Linearization and step.
    buffer<float>(2 * nobs, args.r);
    buffer<float>(12 * nobs, args.Jp);
    buffer<float>(6 * nobs, args.Jx);
    buffer<float>(any_intr ? 2 * np * nobs : 1, args.Jk);
    const std::size_t nd = sys.has_depth ? nobs : 1;
    dr = buffer<float>(nd, args.dr);
    dJp = buffer<float>(6 * nd, args.dJp);
    dJx = buffer<float>(3 * nd, args.dJx);
    dJc = buffer<float>(2 * nd, args.dJc);
    fr = buffer<float>(roff, args.fr);
    fJ = buffer<float>(joff, args.fJ);
    buffer<float>(roff, args.fm);
    diag_f = buffer<float>(sys.nf, args.diag_f);
    g_f = buffer<float>(sys.nf, args.g_f);
    diag_e = buffer<float>(sys.ne, args.diag_e);
    g_e = buffer<float>(sys.ne, args.g_e);
    buffer<float>(sys.nf, args.d2_f);
    buffer<float>(sys.ne, args.d2_e);
    buffer<float>(9 * npts, args.Cinv);
    buffer<float>(sys.pinv_size, args.Pinv);
    buffer<float>(2 * nobs, args.t);
    buffer<float>(nd, args.td);
    buffer<float>(kChunkFloats * nchunks, args.partial);
    buffer<float>(2 * (npts + nfac), args.vals);
    bvec = buffer<float>(sys.nf, args.b);
    x = buffer<float>(sys.nf, args.x);
    buffer<float>(sys.nf, args.res);
    buffer<float>(sys.nf, args.z);
    buffer<float>(sys.nf, args.p);
    buffer<float>(sys.nf, args.q);
    dx_e = buffer<float>(sys.ne, args.dx_e);

    if (direct) setup_direct();
    else {
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
    args.nfac = static_cast<std::uint32_t>(nfac);
    args.np = static_cast<std::uint32_t>(np);
    args.ntiles = static_cast<std::uint32_t>(ntiles);
    args.ntile_chunks = static_cast<std::uint32_t>(ntile_chunks);
    args.has_depth = sys.has_depth ? 1 : 0;
    args.has_intr = any_intr ? 1 : 0;
    args.loss_type = static_cast<std::int32_t>(sys.loss.type);
    args.loss_scale = static_cast<float>(sys.loss.scale);
    args.pivot_floor = kPivotFloor;
    Ptr unused;
    args_buf = buffer<Args>(1, unused, &args);
    state_buf = buffer<State>(1, unused);
    linearize_pso = lib.pipelines.at(std::string("linearize_") + model_suffix(sys.model));
  }

  void setup_direct() {
    const std::size_t nfb = sys.fblocks.size();
    std::vector<int> block_at(sys.nf + 1, -1);  // f offset -> block
    for (std::size_t b = 0; b < nfb; ++b) block_at[static_cast<std::size_t>(sys.fblocks[b].offset)] = static_cast<int>(b);
    // Blocks an observation touches, with their roles.
    auto touched = [&](std::size_t k, int* blocks) {
      blocks[0] = sys.o_pose[k] >= 0 ? block_at[static_cast<std::size_t>(sys.o_pose[k])] : -1;
      blocks[1] = sys.o_intr[k] >= 0 ? block_at[static_cast<std::size_t>(sys.o_intr[k])] : -1;
      blocks[2] = sys.has_depth && sys.o_dw[k] != 0 && sys.o_dep[k] >= 0
                      ? block_at[static_cast<std::size_t>(sys.o_dep[k])]
                      : -1;
    };
    std::map<std::pair<int, int>, std::vector<TileEntry>> lists;
    auto add = [&](int A, int B, TileEntry e) {
      if (A > B) return;
      lists[{A, B}].push_back(e);
    };
    for (std::size_t j = 0; j < npts; ++j) {
      const bool eliminated = sys.point_off[j] >= 0;
      for (std::size_t ka = sys.pt_begin[j]; ka < sys.pt_begin[j + 1]; ++ka) {
        int ba[3];
        touched(ka, ba);
        for (std::size_t kb = sys.pt_begin[j]; kb < sys.pt_begin[j + 1]; ++kb) {
          if (!eliminated && ka != kb) continue;
          int bb[3];
          touched(kb, bb);
          for (std::uint32_t ra = 0; ra < 3; ++ra)
            for (std::uint32_t rb = 0; rb < 3; ++rb) {
              if (ba[ra] < 0 || bb[rb] < 0) continue;
              add(ba[ra], bb[rb], {static_cast<std::uint32_t>(ka), static_cast<std::uint32_t>(kb), ra | (rb << 2)});
            }
        }
      }
    }
    std::size_t f_global = 0;
    for (const auto& fs : sys.fsets) {
      for (std::size_t f = 0; f < fs.count; ++f, ++f_global) {
        for (int sa = 0; sa < fs.nslots; ++sa)
          for (int sb = 0; sb < fs.nslots; ++sb) {
            const int oa = fs.off(f, sa), ob = fs.off(f, sb);
            if (oa < 0 || ob < 0) continue;
            add(block_at[static_cast<std::size_t>(oa)], block_at[static_cast<std::size_t>(ob)],
                {static_cast<std::uint32_t>(f_global), 0,
                 static_cast<std::uint32_t>(sa) | (static_cast<std::uint32_t>(sb) << 2) | 16u});
          }
      }
    }
    for (std::size_t b = 0; b < nfb; ++b) lists[{static_cast<int>(b), static_cast<int>(b)}];  // every diagonal

    std::vector<Tile> tiles;
    std::vector<TileChunk> tchunks;
    std::vector<TileEntry> entries;
    std::size_t nvals = 0;
    // Tiles by column block, then row block: the order of the compressed columns.
    std::vector<std::vector<std::pair<int, std::size_t>>> by_col(nfb);  // (row block, tile)
    for (auto& [ab, list] : lists) {
      Tile T{};
      T.a = static_cast<std::uint32_t>(ab.first);
      T.b = static_cast<std::uint32_t>(ab.second);
      T.c0 = static_cast<std::uint32_t>(tchunks.size());
      const std::size_t e0 = entries.size();
      entries.insert(entries.end(), list.begin(), list.end());
      for (std::size_t e = e0; e < entries.size(); e += kTileChunk)
        tchunks.push_back({static_cast<std::uint32_t>(tiles.size()), static_cast<std::uint32_t>(e),
                           static_cast<std::uint32_t>(std::min(entries.size(), e + kTileChunk))});
      T.c1 = static_cast<std::uint32_t>(tchunks.size());
      T.voff = static_cast<std::uint32_t>(nvals);
      nvals += static_cast<std::size_t>(sys.fblocks[T.a].size * sys.fblocks[T.b].size);
      by_col[T.a].push_back({ab.second, tiles.size()});
      tiles.push_back(T);
      list.clear();
      list.shrink_to_fit();
    }
    ntiles = tiles.size();
    ntile_chunks = tchunks.size();
    buffer<Tile>(tiles.size(), args.tiles, tiles.data());
    buffer<TileChunk>(tchunks.size(), args.tile_chunks, tchunks.data());
    buffer<TileEntry>(entries.size(), args.entries, entries.data());
    buffer<float>(kChunkFloats * tchunks.size(), args.tile_partial);
    tile_vals = buffer<float>(nvals, args.tile_vals);

    // Lower triangle, compressed columns: column c of block A holds rows r >= c of
    // every tile (A, B), B >= A, in increasing B.
    csc_begin.assign(sys.nf + 1, 0);
    for (std::size_t A = 0; A < nfb; ++A) {
      const auto& fa = sys.fblocks[A];
      for (int col = 0; col < fa.size; ++col) {
        const std::size_t c = static_cast<std::size_t>(fa.offset + col);
        for (const auto& [B, ti] : by_col[A]) {
          const Tile& T = tiles[ti];
          const auto& fb = sys.fblocks[static_cast<std::size_t>(B)];
          for (int y = 0; y < fb.size; ++y) {
            const int r = fb.offset + y;
            if (r < static_cast<int>(c)) continue;
            csc_rows.push_back(r);
            csc_src.push_back(T.voff + static_cast<std::uint32_t>(col * fb.size + y));
          }
        }
        csc_begin[c + 1] = static_cast<long>(csc_rows.size());
      }
    }
    csc_vals.assign(csc_rows.size(), 0.0);
    rhs.assign(sys.nf, 0.0);
    if (sys.nf == 0) return;
    SparseSymbolicFactorOptions opts = _SparseDefaultSymbolicFactorOptions;
    symbolic = SparseFactor(SparseFactorizationCholesky, structure(), opts);
    if (symbolic.status != SparseStatusOK) fail("Accelerate symbolic factorization failed", nil);
    have_symbolic = true;
  }

  SparseMatrixStructure structure() {
    SparseMatrixStructure s{};
    s.rowCount = static_cast<int>(sys.nf);
    s.columnCount = static_cast<int>(sys.nf);
    s.columnStarts = csc_begin.data();
    s.rowIndices = csc_rows.data();
    s.attributes.kind = SparseSymmetric;
    s.attributes.triangle = SparseLowerTriangle;
    s.blockSize = 1;
    return s;
  }

  // Factorizes the tiles the GPU formed and solves S dx_f = b into x. False when
  // the system is not positive definite (the step is then invalid).
  bool direct_solve() {
    // Nothing free on the camera side (only points are refined): the reduced system
    // is empty, dx_f has no entries, and Accelerate must not see the empty structure.
    if (sys.nf == 0) return true;
    const float* tv = host(tile_vals);
    for (std::size_t i = 0; i < csc_vals.size(); ++i) csc_vals[i] = static_cast<double>(tv[csc_src[i]]);
    SparseMatrix_Double S{};
    S.structure = structure();
    S.data = csc_vals.data();
    SparseOpaqueFactorization_Double f = SparseFactor(symbolic, S);
    const bool ok = f.status == SparseStatusOK;
    if (ok) {
      const float* bf = host(bvec);
      for (std::size_t c = 0; c < sys.nf; ++c) rhs[c] = static_cast<double>(bf[c]);
      DenseVector_Double v{};
      v.count = static_cast<int>(sys.nf);
      v.data = rhs.data();
      SparseSolve(f, v);
      float* xf = host(x);
      for (std::size_t c = 0; c < sys.nf; ++c) xf[c] = static_cast<float>(rhs[c]);
    }
    SparseCleanup(f);
    return ok;
  }

  State& state() { return *static_cast<State*>(state_buf.contents); }
  float* host(id<MTLBuffer> b) { return static_cast<float*>(b.contents); }

  // ---------------------------------------------------------------------------
  // Encoding.

  struct Encoder {
    Impl& impl;
    id<MTLCommandBuffer> cb;
    id<MTLComputeCommandEncoder> enc;
    explicit Encoder(Impl& i) : impl(i) {
      cb = [impl.queue commandBuffer];
      enc = [cb computeCommandEncoder];
      [enc useResources:impl.resources.data()
                  count:impl.resources.size()
                  usage:MTLResourceUsageRead | MTLResourceUsageWrite];
      [enc setBuffer:impl.args_buf offset:0 atIndex:0];
      [enc setBuffer:impl.state_buf offset:0 atIndex:1];
    }
    void run(const char* kernel, std::size_t n, const Op& op = {}) { run(impl.lib.pipelines.at(kernel), n, op); }
    void run(id<MTLComputePipelineState> pso, std::size_t n, const Op& op = {}) {
      if (n == 0) return;
      [enc setComputePipelineState:pso];
      [enc setBytes:&op length:sizeof(Op) atIndex:2];
      const NSUInteger group = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, 256);
      [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
    }
    // One threadgroup: a deterministic reduction into a state slot.
    void reduce(int mode, Ptr a, Ptr b, std::size_t n, int slot, int iter) {
      Op op;
      op.x = a;
      op.y = b;
      op.n = static_cast<std::uint32_t>(n);
      op.mode = mode;
      op.slot = slot;
      op.iter = iter;
      [enc setComputePipelineState:impl.lib.pipelines.at("reduce")];
      [enc setBytes:&op length:sizeof(Op) atIndex:2];
      [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(kReduceThreads, 1, 1)];
    }
    // One threadgroup per camera-side block.
    void gather(int mode, Ptr vx = 0, Ptr vy = 0, int iter = 0) {
      const std::size_t nfb = impl.sys.fblocks.size();
      if (nfb == 0) return;
      Op op;
      op.x = vx;
      op.y = vy;
      op.mode = mode;
      op.iter = iter;
      [enc setComputePipelineState:impl.lib.pipelines.at("block_gather")];
      [enc setBytes:&op length:sizeof(Op) atIndex:2];
      [enc dispatchThreadgroups:MTLSizeMake(nfb, 1, 1) threadsPerThreadgroup:MTLSizeMake(kGatherThreads, 1, 1)];
    }
    // One threadgroup per tile of the reduced camera system.
    void tiles() {
      if (impl.ntiles == 0) return;
      if (impl.ntile_chunks > 0) {
        [enc setComputePipelineState:impl.lib.pipelines.at("schur_tiles")];
        [enc dispatchThreadgroups:MTLSizeMake(impl.ntile_chunks, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(kTileThreads, 1, 1)];
      }
      run("tile_finish", kChunkFloats * impl.ntiles);
    }
    void finish() {
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.status != MTLCommandBufferStatusCompleted) fail("command buffer failed", cb.error);
    }
  };

  // Squared column norms and the gradient, both sides.
  void encode_colnorm(Encoder& e) {
    e.gather(0);
    e.run("point_colnorm", npts);
  }

  // y = S x for camera-side vectors x, y (inside the conjugate gradients loop).
  void encode_schur(Encoder& e, Ptr vx, Ptr vy, int iter) {
    Op op;
    op.x = vx;
    op.y = vy;
    op.iter = iter;
    op.mode = 1;
    e.run("schur_points", npts, op);
    e.run("factors_times", nfac, op);
    e.gather(1, vx, vy, iter);
  }

  void encode_pcg_iteration(Encoder& e, int i) {
    Op it;
    it.iter = i;
    e.run("pcg_precondition", sys.fblocks.size(), it);
    e.reduce(1, args.res, args.z, sys.nf, kSlotRho, i);
    e.run("pcg_rho", 1, it);
    e.run("pcg_direction", sys.nf, it);
    encode_schur(e, args.p, args.q, i);
    e.reduce(1, args.p, args.q, sys.nf, kSlotPQ, i);
    e.run("pcg_alpha", 1, it);
    const bool reset = i % kResidualResetPeriod == 0;
    Op up = it;
    up.mode = reset ? 1 : 0;
    e.run("pcg_update", sys.nf, up);
    if (reset) {
      encode_schur(e, args.x, args.q, i);
      e.run("pcg_reset", sys.nf, it);
    }
    e.reduce(2, args.x, 0, sys.nf, kSlotQ, i);
    e.run("pcg_test", 1, it);
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
    }
    if (nfac == 0) return;
    sys.linearize_factors();
    float *r = host(fr), *J = host(fJ);
    for (std::size_t s = 0; s < sys.fsets.size(); ++s) {
      const auto& fs = sys.fsets[s];
      for (std::size_t a = 0; a < fs.r.size(); ++a) r[set_roff[s] + a] = static_cast<float>(fs.r[a]);
      for (std::size_t a = 0; a < fs.J.size(); ++a) J[set_joff[s] + a] = static_cast<float>(fs.J[a]);
    }
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
    @autoreleasepool {
      sys.update_bases();
      if (!res_cur_valid) cost_of(sys.cur, res_cur);
      res_cur_valid = true;
      std::memcpy(hres.contents, res_cur.data(), res_cur.size() * sizeof(float));
      upload_parameters();
      host_rows();
      {
        Encoder e(*this);
        e.run(linearize_pso, sys.nobs);
        if (compute_scaling) encode_colnorm(e);
        e.finish();
      }
      if (compute_scaling) {
        // Ceres: scale = 1 / (1 + column norm), from the first Jacobian.
        const float* df = host(diag_f);
        const float* de = host(diag_e);
        for (std::size_t c = 0; c < sys.nf; ++c) sys.f_scale[c] = 1 / (1 + std::sqrt(static_cast<double>(df[c])));
        for (std::size_t c = 0; c < sys.ne; ++c) sys.e_scale[c] = 1 / (1 + std::sqrt(static_cast<double>(de[c])));
        float* fs = host(f_scale);
        float* es = host(e_scale);
        for (std::size_t c = 0; c < sys.nf; ++c) fs[c] = static_cast<float>(sys.f_scale[c]);
        for (std::size_t c = 0; c < sys.ne; ++c) es[c] = static_cast<float>(sys.e_scale[c]);
        sys.have_scaling = true;
      }
      Encoder e(*this);
      if (sys.have_scaling) {
        e.run("scale_observations", sys.nobs);
        e.run("scale_factors", nfac);
      }
      encode_colnorm(e);
      e.finish();
      const float* gf = host(g_f);
      const float* ge = host(g_e);
      double m = 0;
      for (std::size_t c = 0; c < sys.nf; ++c) m = std::max(m, std::abs(gf[c] / sys.f_scale[c]));
      for (std::size_t c = 0; c < sys.ne; ++c) m = std::max(m, std::abs(ge[c] / sys.e_scale[c]));
      gmax = m;
    }
  }

  StepResult compute_step(double radius, double eta, int min_it, int max_it) {
    StepResult out;
    @autoreleasepool {
      State& s = state();
      std::memset(&s, 0, sizeof(State));
      s.eta = static_cast<float>(eta);
      s.min_it = min_it;
      s.max_it = max_it;
      {
        Encoder e(*this);
        Op damp;
        damp.radius = static_cast<float>(radius);
        e.run("damping", sys.nf + sys.ne, damp);
        e.run("point_blocks", npts);
        e.run("rhs_points", npts);
        e.gather(3);
        if (direct) e.tiles();
        e.finish();
      }
      bool iterative = !direct;
      if (direct) {
        // Strategy B: factorize the reduced camera system on the host. Formed in
        // single precision, it can lose definiteness where points have a degenerate
        // depth direction; that step then falls back to conjugate gradients on the
        // implicit system, which tolerates it.
        out.linear_iterations = 1;
        if (s.bad) return out;
        if (!direct_solve()) iterative = true;
      }
      if (iterative) {
        {
          Encoder e(*this);
          Op pre;
          pre.mode = 2;
          e.run("chunk_partial", nchunks, pre);
          e.run("block_finish", sys.fblocks.size(), pre);
          e.run("pcg_vectors_init", sys.nf);
          e.reduce(1, args.b, args.b, sys.nf, kSlotBB, 0);
          e.run("pcg_init", 1);
          e.finish();
        }
        // Conjugate gradients in batches of iterations; after the loop has stopped,
        // the rest of a batch does nothing.
        int i = 1, batch = 4;
        while (!s.done && i <= max_it) {
          Encoder e(*this);
          for (int n = 0; n < batch && i <= max_it; ++n, ++i) encode_pcg_iteration(e, i);
          e.finish();
          batch = std::min(batch * 2, 64);
        }
        out.linear_iterations = s.iterations;
      }
      if (s.failed) return out;
      {
        Encoder e(*this);
        e.run("back_substitute", npts);
        Op fx;
        fx.x = args.x;
        e.run("factors_times", nfac, fx);
        e.run("factor_model", nfac);
        e.reduce(0, args.vals, 0, npts + nfac, kSlotModel, 0);
        e.reduce(0, args.vals + 4 * (npts + nfac), 0, npts + nfac, kSlotModelAbs, 0);
        e.finish();
      }
      out.model_cost_change = -static_cast<double>(s.slot[kSlotModel]);
      const double resolvable = kResolvable * static_cast<double>(s.slot[kSlotModelAbs]);
      out.step_norm = sys.retract(host(x), host(dx_e));
      out.valid = std::isfinite(out.model_cost_change) && out.model_cost_change > 0 && std::isfinite(out.step_norm);
      if (std::isfinite(out.model_cost_change) && std::abs(out.model_cost_change) <= resolvable) {
        out.valid = false;
        out.precision_limit = true;
      }
    }
    return out;
  }
};

MetalBackend::MetalBackend(const Problem& problem, const SolverOptions& options, const Gauge& gauge)
    : impl_(std::make_unique<Impl>(problem, options, gauge)) {}

MetalBackend::~MetalBackend() = default;

double MetalBackend::cost() {
  impl_->res_cur_valid = true;
  return impl_->cost_of(impl_->sys.cur, impl_->res_cur);
}

void MetalBackend::cost_by_type(double* out) { impl_->sys.cost_by_type(out); }

void MetalBackend::linearize(bool compute_scaling) { impl_->linearize(compute_scaling); }

double MetalBackend::gradient_max_norm() const { return impl_->gmax; }

double MetalBackend::parameter_norm() const { return impl_->sys.parameter_norm(); }

StepResult MetalBackend::compute_step(double radius, double eta, int min_linear_iterations,
                                      int max_linear_iterations) {
  return impl_->compute_step(radius, eta, min_linear_iterations, max_linear_iterations);
}

bool MetalBackend::candidate_cost(double* cost) {
  *cost = impl_->cost_of(impl_->sys.cand, impl_->res_cand);
  return std::isfinite(*cost);
}

void MetalBackend::accept_candidate() {
  impl_->sys.cur.swap(impl_->sys.cand);
  impl_->res_cur.swap(impl_->res_cand);
}

void MetalBackend::write_back(Problem& problem) const { impl_->sys.write_back(problem); }

}  // namespace jba::metal
