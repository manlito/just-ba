// Structures and constants shared by the GPU kernels and the host (D96).
//
// One definition, read by three compilers: the host's C++ compiler (the shared host
// logic, backends/gpu), Metal's run-time compiler (embedded after the Metal prelude,
// backends/metal/kernels/prelude.metal) and nvcc (after the CUDA prelude,
// backends/cuda/kernels/prelude.cuh). The preludes define the address-space macros;
// without one, this header is plain C++ and every pointer is a plain pointer of the
// same size, so the layouts agree by construction, but for uint_pair, which is the
// kernels' uint2 and a struct of two uints on the host (same size and alignment).
// Metal also checks the sizes at run time (layout_sizes).
//
// Kept to the subset of C++ that Metal Shading Language accepts: no standard headers,
// no constructors, no default member values.
#pragma once

#ifndef JBA_DEVICE
// Host C++: no address spaces.
#define JBA_DEVICE
#define JBA_CONSTANT_DATA constexpr
#define JBA_ATOMIC_INT int
#endif

namespace jba_gpu {

#if !defined(__METAL_VERSION__)
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;
#endif
#if defined(__METAL_VERSION__) || defined(__CUDACC__)
using uint_pair = uint2;
#else
struct uint_pair {
  uint x, y;
};
#endif

JBA_CONSTANT_DATA int kMaxParams = 8;
JBA_CONSTANT_DATA int kChunkFloats = 64;    // partial sums per camera chunk (an 8 x 8 block at most)
JBA_CONSTANT_DATA uint kGatherThreads = 256;  // block_gather: threads per camera-side block
JBA_CONSTANT_DATA uint kTileThreads = 64;     // schur_tiles: threads per chunk of tile entries
JBA_CONSTANT_DATA uint kReduceThreads = 256;  // reduce: threads of its one group

// Equal to cpu::BlockKind and jba::LossType (checked on the host).
enum BlockKind : int { kPose = 0, kIntrinsics = 1, kSetRigid = 2, kSetScale = 3, kFrame = 4, kDepth = 5 };
enum LossType : int { kNone = 0, kHuber = 1, kCauchy = 2, kGemanMcClure = 3 };

// Reduction results in State::slot.
enum Slot : int { kSlotBB = 0, kSlotRho = 1, kSlotPQ = 2, kSlotQ = 3, kSlotModel = 4, kSlotModelAbs = 5 };

struct FBlockDesc {
  int offset;
  int size;
  int pinv_offset;
  int kind;
};

// A run of a camera-side block's observation list; never splits the observations of
// one point (the preconditioner needs them together).
struct Chunk {
  uint block;
  uint list;  // 0: per-image list, 1: per-intrinsics-block list
  uint q0;
  uint q1;
};

// A camera-side factor (pose prior, gravity, segment, depth prior): rows x cols
// Jacobian at fJ + joff, residual and product at fr / fm + roff.
struct Factor {
  int rows, cols, joff, roff, nslots;
  int off[3];
  int base[3];
  int size[3];
};

// The explicit reduced camera system (strategy B), by tiles: one per pair of
// camera-side blocks (a <= b) that share a point or a factor, plus every diagonal
// block. A tile's entries are the pairs of observations (of one point) or the
// factors that add to it, in a fixed order.
struct Tile {
  uint a, b;    // camera-side blocks
  uint c0, c1;  // its chunks of entries
  uint voff;    // values: size(a) x size(b), row-major
};

// A run of at most kTileChunk entries of one tile, for one group.
struct TileChunk {
  uint tile, e0, e1;
};

// kind 0: observations of one point, in the roles (pose 0, intrinsics 1, depth 2)
// that touch blocks a and b, given by their places in the role's list (img_list for
// pose and depth, ib_list for intrinsics); kind 1: factor ka, slots role_a, role_b.
struct TileEntry {
  uint ka, kb;
  uint info;  // bits 0-1 role_a, 2-3 role_b, 4 kind, 5 one observation, 6 point eliminated
};

struct Args {
  // Observations, point-major.
  JBA_DEVICE const int* o_img;
  JBA_DEVICE const int* o_pt;
  JBA_DEVICE const float* o_xy;
  JBA_DEVICE const float* o_w;
  JBA_DEVICE const int* o_pose;
  JBA_DEVICE const int* o_intr;
  JBA_DEVICE const float* o_dw;  // depth weight, 0 without a reading
  JBA_DEVICE const int* o_dep;
  // Points, images, intrinsics blocks.
  JBA_DEVICE const uint* pt_begin;
  JBA_DEVICE const int* point_off;
  JBA_DEVICE const int* image_block;
  JBA_DEVICE const int* pose_off;
  JBA_DEVICE const int* intr_off;
  JBA_DEVICE const uchar* pose_mask;   // per image: columns that are not variables
  JBA_DEVICE const ushort* intr_mask;  // per block
  JBA_DEVICE const int* basis_of;
  // Camera-side blocks.
  JBA_DEVICE const uint* img_list;
  JBA_DEVICE const uint* ib_list;
  JBA_DEVICE const Chunk* chunks;
  JBA_DEVICE const uint* block_chunks;  // per f-block, into chunks
  JBA_DEVICE const FBlockDesc* fblocks;
  JBA_DEVICE const Factor* factors;
  JBA_DEVICE const uint* fb_fac_begin;  // per f-block, into fb_fac
  JBA_DEVICE const uint_pair* fb_fac;   // (factor, slot)
  // Parameters in single precision, written by the host before each linearization.
  JBA_DEVICE const float* poses;   // 8 per image: q, t + R c, unused
  JBA_DEVICE const float* rc;      // 4 per image: R c, unused
  JBA_DEVICE const float* points;  // 3 per point, minus the centre c
  JBA_DEVICE const float* intr;
  JBA_DEVICE const float* basis;   // 9 per held heading
  JBA_DEVICE const float* f_scale;
  JBA_DEVICE const float* e_scale;
  JBA_DEVICE const float* hres;    // 2 per observation: whitened residual, from double
  // Linearization.
  JBA_DEVICE float* r;
  JBA_DEVICE float* Jp;
  JBA_DEVICE float* Jx;
  JBA_DEVICE float* Jk;
  JBA_DEVICE float* dr;
  JBA_DEVICE float* dJp;
  JBA_DEVICE float* dJx;
  JBA_DEVICE float* dJc;
  JBA_DEVICE float* fr;
  JBA_DEVICE float* fJ;
  JBA_DEVICE float* fm;
  JBA_DEVICE float* diag_f;
  JBA_DEVICE float* g_f;
  JBA_DEVICE float* diag_e;
  JBA_DEVICE float* g_e;
  // Step.
  JBA_DEVICE float* d2_f;
  JBA_DEVICE float* d2_e;
  JBA_DEVICE float* Cinv;
  JBA_DEVICE float* Pinv;
  JBA_DEVICE float* t;
  JBA_DEVICE float* td;
  JBA_DEVICE float* partial;
  JBA_DEVICE float* vals;
  JBA_DEVICE float* b;
  JBA_DEVICE float* x;
  JBA_DEVICE float* res;
  JBA_DEVICE float* z;
  JBA_DEVICE float* p;
  JBA_DEVICE float* q;
  JBA_DEVICE float* dx_e;
  JBA_DEVICE const Tile* tiles;
  JBA_DEVICE const TileChunk* tile_chunks;
  JBA_DEVICE const TileEntry* entries;
  JBA_DEVICE float* tile_partial;  // kChunkFloats per tile chunk
  JBA_DEVICE float* tile_vals;
  // Strategy B: per observation of an eliminated point and camera-side role it has,
  // E = J_f^T J_x (n x 3), and G = E C^-1 in the same layout, per step; by the
  // observation's place in the role's list, so a tile's entries read them nearly in
  // order. Pose (18 floats) from 0, depth (6) from eo_dep, intrinsics (3 np) from eo_intr.
  JBA_DEVICE float* Eo;
  JBA_DEVICE float* Go;
  // Sizes and options.
  uint nobs, npts, nf, ne, nfb, nchunks, nfac, np, ntiles, ntile_chunks;
  uint eo_dep, eo_intr;
  int has_depth, has_intr, loss_type;
  float loss_scale, pivot_floor;
};

struct State {
  float slot[8];  // reduction results
  float rho, rho_prev, pq, alpha, beta, Q0, eta, model_change;
  int min_it, max_it, done, failed, iterations, pad0, pad1, pad2;
  JBA_ATOMIC_INT bad;  // a block that is not positive definite
};

// Per-dispatch choices: which vectors, which mode.
struct Op {
  JBA_DEVICE const float* x;
  JBA_DEVICE float* y;
  uint n;
  int mode;
  int slot;
  int iter;
  float radius;
  float pad;
};

}  // namespace jba_gpu
