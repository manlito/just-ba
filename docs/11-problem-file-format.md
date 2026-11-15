# 11. Problem file format

The on-disk format for problems, used by the CLI and the tests (milestone 0,
decision D48). Reader and writer: `core/src/problem_io.cpp`. Version 1.

COLMAP's model format remains the interchange format for reconstructions
(document 01). This format is for problems: everything a solve needs, including
what COLMAP's format cannot hold (priors, weights, fixed flags, depth, lines).

## Container

A binary file of named, typed arrays. All integers are little-endian. Every
section starts on an 8-byte boundary, so arrays can be mapped or read with
`numpy.frombuffer` directly.

```
header (16 bytes)
  magic        8 bytes   0x89 'J' 'B' 'A' '\r' '\n' 0x1a '\n'
  version      u32       1
  sections     u32       number of sections that follow

section (repeated)
  name_len     u32       at most 256
  name         name_len bytes, UTF-8, no terminator
  dtype        u8        1 u8, 2 i32, 3 i64, 4 f32, 5 f64
  ndim         u8
  reserved     u16       0
  shape        ndim x u64
  padding      zero bytes up to the next multiple of 8
  data         product(shape) x size(dtype) bytes, row-major
  padding      zero bytes up to the next multiple of 8
```

The magic is built like PNG's: the high byte and the line endings catch files
damaged by text-mode transfers. Nothing may follow the last section.

## Rules for readers and writers

- A reader fails with a message on a wrong magic, an unknown version, a truncated
  section, trailing data, a known section with the wrong dtype or shape, or a
  problem that does not validate. It never reads a damaged file silently.
- Sections a reader does not know are kept verbatim and written back unchanged.
  New optional sections can therefore be added without a version change.
- The version changes only when an existing section changes meaning, or when a
  data type is added: a reader fails on a data type it does not know, even in a
  section it would otherwise keep.
- A section name appears once. A reader fails on a second copy, and the writer
  refuses to produce one.
- Section order does not matter. Writers use the order of the table below.

## Sections, version 1

`N` images, `M` points, `K` observations, `B` intrinsics blocks, `P` parameters
per block for the camera model.

| Name | dtype | Shape | Required | Meaning |
|---|---|---|---|---|
| `camera.model` | i32 | [1] | yes | COLMAP model id: 0 SIMPLE_PINHOLE, 1 PINHOLE, 2 SIMPLE_RADIAL, 3 RADIAL, 4 OPENCV, 5 OPENCV_FISHEYE |
| `intrinsics` | f64 | [B, P] | yes | Parameters in COLMAP order (document 02) |
| `intrinsics.free` | u8 | [B] | no | Bit mask of freed groups: 1 focal length, 2 principal point, 4 distortion. Absent: all fixed (D26) |
| `image.pose` | f64 | [N, 7] | yes | World to camera: qw, qx, qy, qz, tx, ty, tz. Unit quaternion. Camera axes x right, y down, z forward |
| `image.intrinsics` | i32 | [N] | if B > 1 | Intrinsics block of each image. Absent: every image uses block 0 |
| `image.fixed` | u8 | [N] | no | 1 holds the pose fixed. Absent: all free |
| `point.xyz` | f64 | [M, 3] | yes | World coordinates |
| `point.fixed` | u8 | [M] | no | 1 holds the point fixed. Absent: all free |
| `obs.image` | i32 | [K] | yes | Image index of each observation |
| `obs.point` | i32 | [K] | yes | Point index of each observation |
| `obs.xy` | f64 | [K, 2] | yes | Keypoint in pixels, in the same pixel convention as the principal point in `intrinsics` |
| `obs.sigma` | f64 | [K] | no | Keypoint standard deviation in pixels. Absent: 1 |

Pose priors (milestone 2; all optional). `R` relative priors, `G` global priors,
`S` prior sets.

| Name | dtype | Shape | Required | Meaning |
|---|---|---|---|---|
| `prior.relative.images` | i32 | [R, 2] | with relative priors | Images i, j |
| `prior.relative.pose` | f64 | [R, 7] | with relative priors | The motion T_j T_i^-1 (camera i to camera j coordinates): qw, qx, qy, qz, tx, ty, tz |
| `prior.relative.sigma` | f64 | [R, 2] | no | Rotation (rad) and translation (m) standard deviations; +inf leaves that part out |
| `prior.relative.sqrt_info` | f64 | [R, 6, 6] | no | Whitening matrix W, residual W r; rotation rows first. Not with sigma |
| `prior.global.image` | i32 | [G] | with global priors | Image of each prior |
| `prior.global.set` | i32 | [G] | no | Prior set of each prior. Absent: all in set 0 |
| `prior.global.pose` | f64 | [G, 7] | with global priors | World to camera in the set's frame, as `image.pose` |
| `prior.global.sigma` | f64 | [G, 2] | no | Rotation (rad) and position (set units) standard deviations; +inf leaves that part out |
| `prior.global.sqrt_info` | f64 | [G, 6, 6] | no | As for relative priors |
| `prior.set.relation` | i32 | [S] | no | 0 the set is the working frame, 1 rigid transform, 2 similarity (D19). Absent: one set, the working frame |
| `prior.set.transform` | f64 | [S, 8] | no | qw, qx, qy, qz, tx, ty, tz, s with X_working = s R X_set + t. Starting value, and the estimate after a solve. Absent: identity |

Without sigma or sqrt_info a prior gets the solver's default weights (D67).

Robust losses (milestone 2; optional, D80). `T` is at most the number of residual
types, indexed in this order: 0 reprojection, 1 relative prior, 2 global prior.
Types appended later take the next indices.

| Name | dtype | Shape | Required | Meaning |
|---|---|---|---|---|
| `loss.type` | i32 | [T] | with `loss.scale` | 0 none, 1 Huber, 2 Cauchy, 3 Geman-McClure (definitions in document 02) |
| `loss.scale` | f64 | [T] | with `loss.type` | Scale a of each loss, in whitened units (pixels for reprojection, standard deviations for priors) |

Absent, or for types past T, the solver options decide. Index 3 is the gravity prior,
4 the Manhattan line, 5 the depth term (milestone 3). The CLI and the parity tool
start from these and let their flags override them; `refine` writes the losses it
solved with; the library's `solve` takes its losses from its options, and
`with_problem_losses` copies these in.
Residual definitions: decisions D64 and D65.

Validation: index ranges, array lengths, finite values, positive sigmas (prior
sigmas may be +inf), known loss types with positive finite scales, and quaternion
and direction norms within 1e-9 of one. A relative prior may not tie an image to itself; a set
that is the working frame must have the identity transform, and a rigid set scale
1.

Normally a session has one intrinsics block (D11). Several blocks, all of the
model in `camera.model`, are allowed so that BAL problems, which have one camera
per image, convert without loss; the solver accepts them too (D53).

Gravity priors (milestone 3; all optional, D82). `Q` gravity priors.

| Name | dtype | Shape | Required | Meaning |
|---|---|---|---|---|
| `gravity.world` | f64 | [3] | no | Direction of gravity in the working frame, unit. Absent: (0, -1, 0), the world of ARKit and ARCore |
| `prior.gravity.image` | i32 | [Q] | with gravity priors | Image of each prior |
| `prior.gravity.direction` | f64 | [Q, 3] | with gravity priors | Direction of gravity measured in the image's camera axes, unit |
| `prior.gravity.sigma` | f64 | [Q] | no | Standard deviation in radians. Absent: the solver's default |

Line segments and the Manhattan frame (milestone 3; all optional, D84 to D86). `L`
segments.

| Name | dtype | Shape | Required | Meaning |
|---|---|---|---|---|
| `lines.image` | i32 | [L] | with segments | Image of each segment |
| `lines.xy` | f64 | [L, 4] | with segments | Endpoints x0, y0, x1, y1 in pixels; not equal |
| `lines.sigma` | f64 | [L] | no | Endpoint standard deviation in pixels. Absent: the solver's default |
| `lines.axis` | i32 | [L] | no | Building axis: 0 and 1 horizontal, 2 vertical, -1 none. Absent: none assigned |
| `manhattan.frame` | f64 | [4] | no | Quaternion from the building's axes to the working frame. Absent: no Manhattan term |
| `manhattan.dof` | i32 | [1] | no | 1: the frame's heading about the vertical only, axis 2 must be up (-`gravity.world`); 3: its whole rotation. Absent: 1 |

The Manhattan term is in the problem when `manhattan.frame` is present and some
segment has an axis.

Depth at keypoints (milestone 3; all optional, D87).

| Name | dtype | Shape | Required | Meaning |
|---|---|---|---|---|
| `depth.obs` | f64 | [K] | no | Depth reading of each observation, NaN for none |
| `depth.obs_sigma` | f64 | [K] | no | Its standard deviation, in the reading's units. Absent: the solver's default |
| `depth.kind` | i32 | [1] | with `depth.obs` | 0 along the optical axis, 1 along the ray |
| `depth.form` | i32 | [1] | with `depth.obs` | 0 linear, 1 inverse |
| `depth.correction` | f64 | [N, 2] | no | Per image (a, b); a d + b is the corrected reading. Absent: (1, 0). The estimate after a solve |
| `depth.correction_free` | u8 | [N] | no | Bits estimated: 1 the scale, 2 the offset. Absent: none (a metric sensor) |

## BAL conversion

`read_bal` and `just-ba convert-bal` turn a Bundle Adjustment in the Large file
into this format, as the RADIAL model with one intrinsics block per image (document
05). BAL cameras look down -z with image y up. Rotating the camera frame by pi
about x gives our convention exactly: `R' = F R`, `t' = F t`, `F = diag(1, -1, -1)`,
and observed y changes sign. The cost is unchanged, which `tests/test_bal.cpp`
checks against BAL's own camera model. Focal length and distortion are marked free
and the principal point fixed, as in BAL.

## Reading from Python

```python
import numpy as np

def read_jba(path):
    data = open(path, "rb").read()
    assert data[:8] == b"\x89JBA\r\n\x1a\n"
    version, count = np.frombuffer(data, "<u4", 2, 8)
    assert version == 1
    dtypes = {1: "<u1", 2: "<i4", 3: "<i8", 4: "<f4", 5: "<f8"}
    pos, out = 16, {}
    for _ in range(count):
        n = int(np.frombuffer(data, "<u4", 1, pos)[0]); pos += 4
        name = data[pos:pos + n].decode(); pos += n
        dtype, ndim = data[pos], data[pos + 1]; pos += 4
        shape = tuple(int(d) for d in np.frombuffer(data, "<u8", ndim, pos)); pos += 8 * ndim
        pos = (pos + 7) // 8 * 8
        a = np.frombuffer(data, dtypes[dtype], int(np.prod(shape)), pos).reshape(shape)
        pos = (pos + a.nbytes + 7) // 8 * 8
        out[name] = a
    return out
```

The Python package (milestone 2) will ship a reader and writer; until then this
snippet is the reference.
