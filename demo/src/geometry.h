// Small rotation helpers for the demo. Poses are world to camera, qw qx qy qz tx ty tz,
// as in the library and in poselib-gpu. Matrices are row-major 3 x 3.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace demo {

inline constexpr double kPi = 3.14159265358979323846;

inline void quat_to_rotmat(const double* q, double R[9]) {
  const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  const double w = q[0] / n, x = q[1] / n, y = q[2] / n, z = q[3] / n;
  R[0] = 1 - 2 * (y * y + z * z);
  R[1] = 2 * (x * y - w * z);
  R[2] = 2 * (x * z + w * y);
  R[3] = 2 * (x * y + w * z);
  R[4] = 1 - 2 * (x * x + z * z);
  R[5] = 2 * (y * z - w * x);
  R[6] = 2 * (x * z - w * y);
  R[7] = 2 * (y * z + w * x);
  R[8] = 1 - 2 * (x * x + y * y);
}

inline void rotmat_to_quat(const double R[9], double* q) {
  const double tr = R[0] + R[4] + R[8];
  if (tr > 0) {
    const double s = 2 * std::sqrt(tr + 1);
    q[0] = 0.25 * s;
    q[1] = (R[7] - R[5]) / s;
    q[2] = (R[2] - R[6]) / s;
    q[3] = (R[3] - R[1]) / s;
  } else if (R[0] > R[4] && R[0] > R[8]) {
    const double s = 2 * std::sqrt(1 + R[0] - R[4] - R[8]);
    q[0] = (R[7] - R[5]) / s;
    q[1] = 0.25 * s;
    q[2] = (R[1] + R[3]) / s;
    q[3] = (R[2] + R[6]) / s;
  } else if (R[4] > R[8]) {
    const double s = 2 * std::sqrt(1 + R[4] - R[0] - R[8]);
    q[0] = (R[2] - R[6]) / s;
    q[1] = (R[1] + R[3]) / s;
    q[2] = 0.25 * s;
    q[3] = (R[5] + R[7]) / s;
  } else {
    const double s = 2 * std::sqrt(1 + R[8] - R[0] - R[4]);
    q[0] = (R[3] - R[1]) / s;
    q[1] = (R[2] + R[6]) / s;
    q[2] = (R[5] + R[7]) / s;
    q[3] = 0.25 * s;
  }
  if (q[0] < 0)
    for (int k = 0; k < 4; ++k) q[k] = -q[k];
}

inline void mat3_mul(const double A[9], const double B[9], double C[9]) {
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) C[3 * r + c] = A[3 * r] * B[c] + A[3 * r + 1] * B[3 + c] + A[3 * r + 2] * B[6 + c];
}

inline void transpose3(const double A[9], double T[9]) {
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) T[3 * r + c] = A[3 * c + r];
}

// Y = R X + t
inline void apply(const double R[9], const double t[3], const double X[3], double Y[3]) {
  for (int r = 0; r < 3; ++r) Y[r] = R[3 * r] * X[0] + R[3 * r + 1] * X[1] + R[3 * r + 2] * X[2] + t[r];
}

// The camera centre of a world-to-camera pose: -R' t.
inline void camera_centre(const double* pose, double C[3]) {
  double R[9];
  quat_to_rotmat(pose, R);
  for (int c = 0; c < 3; ++c) C[c] = -(R[c] * pose[4] + R[3 + c] * pose[5] + R[6 + c] * pose[6]);
}

// Angle in degrees between two rotation matrices.
inline double rotation_angle_deg(const double A[9], const double B[9]) {
  double Bt[9], D[9];
  transpose3(B, Bt);
  mat3_mul(A, Bt, D);
  return std::acos(std::clamp(0.5 * (D[0] + D[4] + D[8] - 1.0), -1.0, 1.0)) * 180.0 / kPi;
}

// splitmix64: deterministic draws for the generated scenes, the same on every platform.
class Rng {
 public:
  explicit Rng(std::uint64_t seed) : s_(seed * 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull) {}
  std::uint64_t next() {
    std::uint64_t z = (s_ += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
  double uniform(double a, double b) { return a + (b - a) * uniform(); }
  double normal() {
    const double u = std::max(uniform(), 1e-300), v = uniform();
    return std::sqrt(-2.0 * std::log(u)) * std::cos(2.0 * kPi * v);
  }

 private:
  std::uint64_t s_;
};

// A rotation by up to max_deg about a random axis.
inline void random_rotation(Rng& rng, double max_deg, double R[9]) {
  double a[3] = {rng.normal(), rng.normal(), rng.normal()};
  const double n = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  const double half = 0.5 * rng.uniform(0.0, max_deg) * kPi / 180.0;
  const double q[4] = {std::cos(half), std::sin(half) * a[0] / n, std::sin(half) * a[1] / n, std::sin(half) * a[2] / n};
  quat_to_rotmat(q, R);
}

}  // namespace demo
