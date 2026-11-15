#pragma once

#include <cmath>

// Small fixed-size rotation helpers. Quaternions are (w, x, y, z), unit norm,
// Hamilton convention, and rotate a vector as q * v * conj(q).

namespace jba {

template <typename T>
void quat_multiply(const T* a, const T* b, T* out) {
  const T w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
  const T x = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
  const T y = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
  const T z = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
  out[0] = w;
  out[1] = x;
  out[2] = y;
  out[3] = z;
}

// Quaternion of the rotation vector `v` (axis times angle, radians).
template <typename T>
void quat_exp(const T* v, T* q) {
  using std::cos;
  using std::sin;
  using std::sqrt;
  const T theta2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  const T theta = sqrt(theta2);
  T s;
  if (theta2 > T(1e-12)) {
    s = sin(theta / T(2)) / theta;
    q[0] = cos(theta / T(2));
  } else {
    // Taylor expansion near zero.
    s = T(0.5) - theta2 / T(48);
    q[0] = T(1) - theta2 / T(8);
  }
  q[1] = s * v[0];
  q[2] = s * v[1];
  q[3] = s * v[2];
}

template <typename T>
void quat_normalize(T* q) {
  using std::sqrt;
  const T n = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  for (int i = 0; i < 4; ++i) q[i] /= n;
}

template <typename T>
void quat_rotate(const T* q, const T* v, T* out) {
  // t = 2 * cross(q.xyz, v); out = v + w * t + cross(q.xyz, t)
  const T tx = T(2) * (q[2] * v[2] - q[3] * v[1]);
  const T ty = T(2) * (q[3] * v[0] - q[1] * v[2]);
  const T tz = T(2) * (q[1] * v[1] - q[2] * v[0]);
  out[0] = v[0] + q[0] * tx + (q[2] * tz - q[3] * ty);
  out[1] = v[1] + q[0] * ty + (q[3] * tx - q[1] * tz);
  out[2] = v[2] + q[0] * tz + (q[1] * ty - q[2] * tx);
}

// Applies the tangent step of a pose (decision D46): R <- Exp(dtheta) * R,
// t <- t + dt. `pose` is qw qx qy qz tx ty tz, `step` is dtheta (3) then dt (3).
template <typename T>
void pose_retract(const T* pose, const T* step, T* out) {
  T dq[4];
  quat_exp(step, dq);
  quat_multiply(dq, pose, out);
  quat_normalize(out);
  out[4] = pose[4] + step[3];
  out[5] = pose[5] + step[4];
  out[6] = pose[6] + step[5];
}

}  // namespace jba
