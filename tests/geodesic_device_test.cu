// CUDA compile-and-run smoke test for the CPU/GPU-shared geodesic integrator
// (include/blackhole/geodesic.hpp). It checks the "header is usable from device
// code" requirement: a __global__ kernel calls the Vec3 operators, makeRay,
// acceleration, stepVerlet, stepRk4, adaptiveStep and trace, and the test then
//   (a) compares every device result with the host result of the very same
//       calls (same BH_HD function, run on the CPU), and
//   (b) compares the device results with independent analytic / double
//       precision values, so a header that is wrong (but identically wrong on
//       host and device) still fails.
//
// Part of the focused geodesic filter: --gtest_filter='*Geodesic*'.
// Skipped (not failed) when no CUDA device is available.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>

#include "blackhole/geodesic.hpp"
#include "cuda_test_support.hpp"

#ifndef BH_HD
#error "geodesic.hpp must define the BH_HD host/device macro"
#endif

namespace {

using namespace blackhole::geo;

// Plain-old-data record of everything the shared routine computes.
struct SmokeResult {
  // Vec3 operators on a = (1,2,3), b = (-4,5,0.5)
  Vec3 sum, diff, scaledL, scaledR, crossAB, unit;
  float dotAB, lenPythagoras;  // length({3,4,12}) == 13
  // makeRay({3,0,0}, {0,2,0})
  Ray ray0;
  // acceleration({2,0,0}, 3)
  Vec3 acc;
  // one step of dt = 0.25 from ray0
  Ray verlet, rk4;
  // adaptiveStep
  float dtFar, dtNear, dtClampLow;
  // trace(): ray from (-30,0,0) with b = 2 b_c (escapes) and b = 0.5 b_c
  // (captured), default TraceParams.
  TraceResult escaped, captured;
};

constexpr float kStartDistance = 30.0f;

// Ray from (-30, 0, 0) with impact parameter exactly b (direction
// (sqrt(D^2 - b^2), b, 0); makeRay normalises it).
BH_HD Ray rayWithImpact(float b) {
  return makeRay({-kStartDistance, 0.0f, 0.0f},
                 {sqrtf(kStartDistance * kStartDistance - b * b), b, 0.0f});
}

// The shared routine: compiled for both host and device.
BH_HD void runSmoke(SmokeResult* out) {
  const Vec3 a{1.0f, 2.0f, 3.0f};
  const Vec3 b{-4.0f, 5.0f, 0.5f};
  out->sum = a + b;
  out->diff = a - b;
  out->scaledL = a * 2.0f;
  out->scaledR = 2.0f * a;
  out->dotAB = dot(a, b);
  out->crossAB = cross(a, b);
  out->lenPythagoras = length({3.0f, 4.0f, 12.0f});
  out->unit = normalize({0.0f, 3.0f, 4.0f});

  out->ray0 = makeRay({3.0f, 0.0f, 0.0f}, {0.0f, 2.0f, 0.0f});
  out->acc = acceleration({2.0f, 0.0f, 0.0f}, 3.0f);

  out->verlet = out->ray0;
  stepVerlet(out->verlet, 0.25f);
  out->rk4 = out->ray0;
  stepRk4(out->rk4, 0.25f);

  out->dtFar = adaptiveStep(10.0f, false);
  out->dtNear = adaptiveStep(3.0f, true);
  out->dtClampLow = adaptiveStep(0.0f, false);

  const TraceParams params;  // rEscape = 40, maxSteps = 2000
  out->escaped = trace(rayWithImpact(2.0f * kCriticalImpact), params);
  out->captured = trace(rayWithImpact(0.5f * kCriticalImpact), params);
}

__global__ void smokeKernel(SmokeResult* out) { runSmoke(out); }

// --- independent double-precision references --------------------------------

struct D3 {
  double x, y, z;
};
D3 operator+(D3 a, D3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
D3 operator*(D3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
double dotD(D3 a, D3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
D3 accelD(D3 p, double h2) {
  const double r = std::sqrt(dotD(p, p));
  return p * (-1.5 * h2 / (r * r * r * r * r));
}

void expectNear(const Vec3& got, const Vec3& want, double tol,
                const char* what) {
  EXPECT_NEAR(got.x, want.x, tol) << what << ".x";
  EXPECT_NEAR(got.y, want.y, tol) << what << ".y";
  EXPECT_NEAR(got.z, want.z, tol) << what << ".z";
}

void expectRayNear(const Ray& got, const Ray& want, double tol,
                   const char* what) {
  expectNear(got.pos, want.pos, tol, what);
  expectNear(got.vel, want.vel, tol, what);
  EXPECT_NEAR(got.h2, want.h2, tol) << what << ".h2";
}

TEST(GeodesicDeviceTest, SharedHeaderRunsOnDeviceAndMatchesHost) {
  SKIP_IF_NO_CUDA_DEVICE();

  SmokeResult host{};
  runSmoke(&host);

  SmokeResult* dev = nullptr;
  ASSERT_EQ(cudaMalloc(&dev, sizeof(SmokeResult)), cudaSuccess);
  ASSERT_EQ(cudaMemset(dev, 0, sizeof(SmokeResult)), cudaSuccess);
  smokeKernel<<<1, 1>>>(dev);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  SmokeResult d{};
  ASSERT_EQ(cudaMemcpy(&d, dev, sizeof(SmokeResult), cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(dev);

  // ---- (a) device == host ----------------------------------------------------
  // Device code contracts a*b+c into FMAs, the host build may not; for the
  // single-step quantities (magnitudes <= 13, <= ~10 float operations) that is
  // a few ulps (measured max 1.2e-7 with a correct implementation). A
  // tolerance of 1e-5 keeps ~80x headroom and is still
  // 3 orders of magnitude below the physics effect (~0.03 velocity change per
  // step at r = 3).
  constexpr double kStepTol = 1e-5;
  expectNear(d.sum, host.sum, kStepTol, "sum");
  expectNear(d.diff, host.diff, kStepTol, "diff");
  expectNear(d.scaledL, host.scaledL, kStepTol, "scaledL");
  expectNear(d.scaledR, host.scaledR, kStepTol, "scaledR");
  expectNear(d.crossAB, host.crossAB, kStepTol, "cross");
  expectNear(d.unit, host.unit, kStepTol, "normalize");
  EXPECT_NEAR(d.dotAB, host.dotAB, kStepTol);
  EXPECT_NEAR(d.lenPythagoras, host.lenPythagoras, kStepTol);
  expectRayNear(d.ray0, host.ray0, kStepTol, "makeRay");
  expectNear(d.acc, host.acc, kStepTol, "acceleration");
  expectRayNear(d.verlet, host.verlet, kStepTol, "stepVerlet");
  expectRayNear(d.rk4, host.rk4, kStepTol, "stepRk4");
  EXPECT_FLOAT_EQ(d.dtFar, host.dtFar);
  EXPECT_FLOAT_EQ(d.dtNear, host.dtNear);
  EXPECT_FLOAT_EQ(d.dtClampLow, host.dtClampLow);

  // trace(): ~50..100 float steps; FMA differences accumulate to ~1e-5, so
  // 1e-3 absolute on positions of magnitude ~40 leaves >= 200x headroom
  // (measured max device-host difference 4.8e-6). The
  // outcome and step count are discrete and must agree exactly (neither ray is
  // near a decision boundary: b = 2 b_c and 0.5 b_c).
  auto compareTrace = [](const TraceResult& dr, const TraceResult& hr,
                         const char* name) {
    EXPECT_EQ(dr.outcome, hr.outcome) << name;
    EXPECT_EQ(dr.steps, hr.steps) << name;
    expectRayNear(dr.ray, hr.ray, 1e-3, name);
  };
  compareTrace(d.escaped, host.escaped, "escaped");
  compareTrace(d.captured, host.captured, "captured");

  // ---- (b) device == independent analytic / double references -----------------
  // Vec3 operators (exact small-integer arithmetic).
  expectNear(d.sum, {-3.0f, 7.0f, 3.5f}, 1e-6, "sum");
  expectNear(d.diff, {5.0f, -3.0f, 2.5f}, 1e-6, "diff");
  expectNear(d.scaledL, {2.0f, 4.0f, 6.0f}, 1e-6, "scaledL");
  expectNear(d.scaledR, {2.0f, 4.0f, 6.0f}, 1e-6, "scaledR");
  EXPECT_NEAR(d.dotAB, 7.5f, 1e-6);
  expectNear(d.crossAB, {-14.0f, -12.5f, 13.0f}, 1e-5, "cross");
  EXPECT_NEAR(d.lenPythagoras, 13.0f, 1e-5);
  expectNear(d.unit, {0.0f, 0.6f, 0.8f}, 1e-6, "normalize");

  // makeRay({3,0,0}, {0,2,0}): unit velocity along +y, h = 3 => h2 = 9.
  expectNear(d.ray0.pos, {3.0f, 0.0f, 0.0f}, 1e-6, "ray0.pos");
  expectNear(d.ray0.vel, {0.0f, 1.0f, 0.0f}, 1e-6, "ray0.vel");
  EXPECT_NEAR(d.ray0.h2, 9.0f, 1e-5);

  // a = -1.5 * 3 * (2,0,0) / 2^5 = (-0.28125, 0, 0).
  expectNear(d.acc, {-0.28125f, 0.0f, 0.0f}, 1e-6, "acc");

  // One Verlet / RK4 step of dt = 0.25 from ray0 in double precision.
  {
    const double dt = 0.25, h2 = 9.0;
    const D3 x0{3, 0, 0}, v0{0, 1, 0};
    const D3 a0 = accelD(x0, h2);
    const D3 x1 = x0 + v0 * dt + a0 * (0.5 * dt * dt);
    const D3 v1 = v0 + (a0 + accelD(x1, h2)) * (0.5 * dt);
    EXPECT_NEAR(d.verlet.pos.x, x1.x, kStepTol);
    EXPECT_NEAR(d.verlet.pos.y, x1.y, kStepTol);
    EXPECT_NEAR(d.verlet.vel.x, v1.x, kStepTol);
    EXPECT_NEAR(d.verlet.vel.y, v1.y, kStepTol);
    EXPECT_NEAR(d.verlet.h2, h2, kStepTol);

    const D3 k1p = v0, k1v = a0;
    const D3 k2p = v0 + k1v * (dt / 2), k2v = accelD(x0 + k1p * (dt / 2), h2);
    const D3 k3p = v0 + k2v * (dt / 2), k3v = accelD(x0 + k2p * (dt / 2), h2);
    const D3 k4p = v0 + k3v * dt, k4v = accelD(x0 + k3p * dt, h2);
    const D3 xr = x0 + (k1p + k2p * 2 + k3p * 2 + k4p) * (dt / 6);
    const D3 vr = v0 + (k1v + k2v * 2 + k3v * 2 + k4v) * (dt / 6);
    EXPECT_NEAR(d.rk4.pos.x, xr.x, kStepTol);
    EXPECT_NEAR(d.rk4.pos.y, xr.y, kStepTol);
    EXPECT_NEAR(d.rk4.vel.x, vr.x, kStepTol);
    EXPECT_NEAR(d.rk4.vel.y, vr.y, kStepTol);
    EXPECT_NEAR(d.rk4.h2, h2, kStepTol);
  }

  // adaptiveStep: 0.12 r clamped to [0.02, 0.7], capped at 0.085 near the disk.
  EXPECT_NEAR(d.dtFar, 0.7f, 1e-6);
  EXPECT_NEAR(d.dtNear, 0.085f, 1e-6);
  EXPECT_NEAR(d.dtClampLow, 0.02f, 1e-6);

  // trace(): b = 2 b_c escapes, b = 0.5 b_c is captured, with the step budget
  // not exhausted and h2 carried through unchanged.
  const Ray escStart = rayWithImpact(2.0f * kCriticalImpact);
  EXPECT_EQ(d.escaped.outcome, Outcome::Escaped);
  EXPECT_GT(d.escaped.steps, 0);
  EXPECT_LT(d.escaped.steps, 2000);
  EXPECT_GE(length(d.escaped.ray.pos), 40.0f);
  EXPECT_GT(dot(d.escaped.ray.pos, d.escaped.ray.vel), 0.0f);
  EXPECT_NEAR(d.escaped.ray.h2, escStart.h2, 1e-4);
  // Gravity bent the ray towards -y (it started climbing in +y).
  EXPECT_LT(d.escaped.ray.vel.y, escStart.vel.y);

  EXPECT_EQ(d.captured.outcome, Outcome::Horizon);
  EXPECT_GT(d.captured.steps, 0);
  EXPECT_LT(d.captured.steps, 2000);
  EXPECT_LE(length(d.captured.ray.pos), kRs);
}

}  // namespace
