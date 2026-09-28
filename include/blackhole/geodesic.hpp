// Header-only, CPU/GPU-shared Schwarzschild null-geodesic integrator.
//
// Units: rs = 1, c = 1 (so M = rs / 2). Photons follow the exact Schwarzschild
// orbit equation u'' + u = 1.5 rs u^2 (Binet form), written in pseudo-Cartesian
// coordinates as a central force
//
//     r'' = -1.5 * rs * h2 * r_vec / |r|^5,      h2 = |r_vec x v|^2 (const).
//
// h2 is conserved. The speed is NOT: the energy E = |v|^2 - rs * h2 / |r|^3 is
// conserved instead, so |v| == 1 only at ray creation (and far away).
#pragma once

namespace blackhole {
namespace geo {

#ifndef BH_HD
#if defined(__CUDACC__)
#define BH_HD __host__ __device__
#else
#define BH_HD
#endif
#endif

struct Vec3 {
  float x, y, z;
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
BH_HD inline Vec3 operator*(float s, Vec3 a) { return a * s; }
BH_HD inline float dot(Vec3 a, Vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
BH_HD inline Vec3 cross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
BH_HD inline float length(Vec3 a) { return __builtin_sqrtf(dot(a, a)); }
BH_HD inline Vec3 normalize(Vec3 a) { return a * (1.0f / length(a)); }

inline constexpr float kRs = 1.0f;
inline constexpr float kPhotonSphere = 1.5f;
inline constexpr float kIsco = 3.0f;
inline constexpr float kCriticalImpact = 2.598076211f;  // 1.5 * sqrt(3)

struct Ray {
  Vec3 pos;
  Vec3 vel;
  float h2;
};

BH_HD inline Ray makeRay(Vec3 pos, Vec3 dir) {
  const Vec3 v = normalize(dir);
  const Vec3 h = cross(pos, v);
  return {pos, v, dot(h, h)};
}

// Photon acceleration for angular momentum^2 h2 at position pos.
BH_HD inline Vec3 acceleration(Vec3 pos, float h2) {
  const float r2 = dot(pos, pos);
  const float r = __builtin_sqrtf(r2);
  return pos * (-1.5f * kRs * h2 / (r2 * r2 * r));
}

// Velocity Verlet (2nd order). h2 is left untouched.
BH_HD inline void stepVerlet(Ray& r, float dt) {
  const Vec3 a0 = acceleration(r.pos, r.h2);
  r.pos = r.pos + r.vel * dt + a0 * (0.5f * dt * dt);
  const Vec3 a1 = acceleration(r.pos, r.h2);
  r.vel = r.vel + (a0 + a1) * (0.5f * dt);
}

// Classic RK4 (4th order) on (pos, vel). h2 is left untouched.
BH_HD inline void stepRk4(Ray& r, float dt) {
  const float h2 = r.h2;
  const float half = 0.5f * dt;
  const Vec3 k1p = r.vel;
  const Vec3 k1v = acceleration(r.pos, h2);
  const Vec3 k2p = r.vel + k1v * half;
  const Vec3 k2v = acceleration(r.pos + k1p * half, h2);
  const Vec3 k3p = r.vel + k2v * half;
  const Vec3 k3v = acceleration(r.pos + k2p * half, h2);
  const Vec3 k4p = r.vel + k3v * dt;
  const Vec3 k4v = acceleration(r.pos + k3p * dt, h2);
  r.pos = r.pos + (k1p + k2p * 2.0f + k3p * 2.0f + k4p) * (dt / 6.0f);
  r.vel = r.vel + (k1v + k2v * 2.0f + k3v * 2.0f + k4v) * (dt / 6.0f);
}

BH_HD inline float adaptiveStep(float r, bool nearDisk) {
  float dt = 0.12f * r;
  dt = dt < 0.02f ? 0.02f : (dt > 0.7f ? 0.7f : dt);
  if (nearDisk && dt > 0.085f) dt = 0.085f;
  return dt;
}

// True where the fine disk-plane step cap applies: within 1 unit of the plane
// y = 0 and inside the disk's outer radius (plus a margin of 1).
BH_HD inline bool nearDiskPlane(const Vec3& pos, float diskOuter) {
  return __builtin_fabsf(pos.y) < 1.0f && length(pos) <= diskOuter + 1.0f;
}

enum class Outcome { Horizon, Escaped, MaxSteps };

struct TraceParams {
  float rEscape = 40.f;
  int maxSteps = 2000;
  bool nearDisk = false;
};

struct TraceResult {
  Outcome outcome;
  Ray ray;
  int steps;
};

// Integrates with adaptive-step Verlet until the ray is captured
// (|pos| <= rs), escapes (|pos| >= rEscape and moving outward) or maxSteps is
// exhausted. `steps` counts the steps taken; the capture/escape test runs after
// each step, so a ray that starts inside the horizon is captured at steps == 1.
BH_HD inline TraceResult trace(Ray r, TraceParams p) {
  for (int i = 1; i <= p.maxSteps; ++i) {
    stepVerlet(r, adaptiveStep(length(r.pos), p.nearDisk));
    const float rad = length(r.pos);
    if (rad <= kRs) return {Outcome::Horizon, r, i};
    if (rad >= p.rEscape && dot(r.pos, r.vel) > 0.0f) {
      return {Outcome::Escaped, r, i};
    }
  }
  return {Outcome::MaxSteps, r, p.maxSteps};
}

}  // namespace geo
}  // namespace blackhole
