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

Validation: index ranges, array lengths, finite values, positive sigmas, and
quaternion norms within 1e-9 of one.

Normally a session has one intrinsics block (D11). Several blocks, all of the
model in `camera.model`, are allowed so that BAL problems, which have one camera
per image, convert without loss; the solver accepts them too (D53).

## Planned sections

Reserved names, added in the milestones that need them. Each will be optional.

| Prefix | Milestone | Content |
|---|---|---|
| `prior.relative.*` | 2 | Image pairs, relative poses, weights |
| `prior.global.*` | 2 | Poses per image, frame relation (D19), weights |
| `image.gravity` | 3 | Gravity direction per image |
| `depth.*` | 3 | Depth at keypoints, confidence, per-image scale and offset |
| `lines.*` | 3 | Line segments per image for the Manhattan prior |
| `loss.*` | 2 | Robust loss per residual type. Planned for milestone 1 and not built; until then the loss is a solver option |

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
