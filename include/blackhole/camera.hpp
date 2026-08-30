#pragma once

// Pinhole camera and primary rays for the flat-space (Phase 1) renderer.
// Header-only and host/device so the CUDA kernel and CPU tests share one
// implementation. The black hole sits at the origin; the disk plane is y = 0.

#include <cmath>

#ifndef BH_HD
#if defined(__CUDACC__)
#define BH_HD __host__ __device__
#else
#define BH_HD
#endif
#endif

namespace blackhole::cam {

struct Vec3 {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

BH_HD inline Vec3 operator+(Vec3 a, Vec3 b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
BH_HD inline Vec3 operator-(Vec3 a, Vec3 b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
BH_HD inline Vec3 operator*(Vec3 a, float s) {
  return {a.x * s, a.y * s, a.z * s};
}
BH_HD inline float dot(Vec3 a, Vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
BH_HD inline Vec3 cross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
          a.x * b.y - a.y * b.x};
}
BH_HD inline float length(Vec3 a) { return sqrtf(dot(a, a)); }
BH_HD inline Vec3 normalize(Vec3 a) { return a * (1.0f / length(a)); }

struct Camera {
  Vec3 pos;
  Vec3 forward;
  Vec3 right;
  Vec3 up;
  float tanHalfFovY = 0.0f;
  float aspect = 0.0f;
  int width = 0;
  int height = 0;
};

/// Camera at distance `distance` from the origin, `elevationDegrees` above the
/// disk plane (y = 0), on the -z side, looking at the origin.
BH_HD inline Camera makeCamera(int width, int height, float distance,
                               float fovYDegrees, float elevationDegrees) {
  constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
  const float el = elevationDegrees * kDegToRad;
  Camera c;
  c.pos = Vec3{0.0f, distance * sinf(el), -distance * cosf(el)};
  c.forward = normalize(c.pos * -1.0f);
  c.right = normalize(cross(Vec3{0.0f, 1.0f, 0.0f}, c.forward));
  c.up = cross(c.forward, c.right);
  c.tanHalfFovY = tanf(fovYDegrees * kDegToRad * 0.5f);
  c.aspect = static_cast<float>(width) / static_cast<float>(height);
  c.width = width;
  c.height = height;
  return c;
}

/// Ray direction for continuous pixel coordinates (pixel centres at +0.5,
/// row 0 at the top).
BH_HD inline Vec3 primaryRayDirSub(const Camera& cam, float fx, float fy) {
  const float ndcX = 2.0f * fx / static_cast<float>(cam.width) - 1.0f;
  const float ndcY = 1.0f - 2.0f * fy / static_cast<float>(cam.height);
  return normalize(cam.forward +
                   cam.right * (ndcX * cam.aspect * cam.tanHalfFovY) +
                   cam.up * (ndcY * cam.tanHalfFovY));
}

BH_HD inline Vec3 primaryRayDir(const Camera& cam, int px, int py) {
  return primaryRayDirSub(cam, static_cast<float>(px) + 0.5f,
                          static_cast<float>(py) + 0.5f);
}

/// Background colour for a ray direction: 0.5 + 0.5 * dir.
BH_HD inline Vec3 skyGradient(Vec3 dir) {
  return Vec3{0.5f + 0.5f * dir.x, 0.5f + 0.5f * dir.y, 0.5f + 0.5f * dir.z};
}

}  // namespace blackhole::cam
