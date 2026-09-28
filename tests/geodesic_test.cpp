// Tests for the header-only CPU/GPU-shared Schwarzschild geodesic integrator
// (include/blackhole/geodesic.hpp). This file is CPU only; the device side is
// covered by tests/geodesic_device_test.cu.
//
// Run the whole geodesic suite (host + device smoke) with the focused filter
//
//     ./build/tests/blackhole_tests --gtest_filter='*Geodesic*'
//
// Units: rs = 1, c = 1 (so M = rs / 2 = 0.5). Photons follow
//
//     r'' = -1.5 * rs * h2 * r_vec / |r|^5,      h2 = |r_vec x v|^2 (const),
//
// i.e. the exact Schwarzschild photon orbit equation u'' + u = 1.5 rs u^2
// written in pseudo-Cartesian coordinates. Two facts used throughout:
//
//   * h = |pos x vel| is conserved (central force).
//   * The speed is NOT constant: the "energy" E = |vel|^2 - rs * h2 / |pos|^3
//     is conserved instead (potential U = -rs h2 / (2 r^3)), so |vel| grows
//     to sqrt(1 + rs h2 / r^3) near the hole. |vel| == 1 only far away.
//
// Float round-off note: everything is single precision, exactly like the GPU
// kernels. Tolerances below are chosen so that truncation/physics effects
// dominate float noise by >= 100x and each test still has >= 3x headroom over
// the error of a correct implementation measured with a reference build.

#include "blackhole/geodesic.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

#ifndef BH_HD
#error "geodesic.hpp must define the BH_HD host/device macro"
#endif

namespace {

using namespace blackhole::geo;

constexpr double kPi = 3.14159265358979323846;

// --- helpers ----------------------------------------------------------------

// Ray starting at (-D, 0, 0) aimed so that the impact parameter is exactly b:
// b = D * sin(alpha), alpha = angle between the direction and the inward
// radial direction (+x). The ray passes on the +y side of the hole, so
// gravity bends it towards -y.
Ray impactRay(float D, float b) {
  const float alpha = std::asin(b / D);
  return makeRay({-D, 0.0f, 0.0f}, {std::cos(alpha), std::sin(alpha), 0.0f});
}

// Angle between the initial and final velocity directions (radians).
double deflectionAngle(const Ray& start, const Ray& end) {
  const Vec3 c = cross(start.vel, end.vel);
  return std::atan2(static_cast<double>(length(c)),
                    static_cast<double>(dot(start.vel, end.vel)));
}

// Conserved energy E = |v|^2 - rs * h2 / r^3.
double energy(const Ray& r) {
  const double rr = length(r.pos);
  return static_cast<double>(dot(r.vel, r.vel)) -
         static_cast<double>(kRs) * r.h2 / (rr * rr * rr);
}

// Independent double-precision reference integrator (classic RK4) of the same
// ODE, written without using any of the code under test.
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
void rk4D(D3& p, D3& v, double h2, double dt) {
  const D3 k1p = v, k1v = accelD(p, h2);
  const D3 k2p = v + k1v * (dt / 2), k2v = accelD(p + k1p * (dt / 2), h2);
  const D3 k3p = v + k2v * (dt / 2), k3v = accelD(p + k2p * (dt / 2), h2);
  const D3 k4p = v + k3v * dt, k4v = accelD(p + k3p * dt, h2);
  p = p + (k1p + k2p * 2 + k3p * 2 + k4p) * (dt / 6);
  v = v + (k1v + k2v * 2 + k3v * 2 + k4v) * (dt / 6);
}
D3 toD(Vec3 a) { return {a.x, a.y, a.z}; }

// Trace with a far escape radius and a generous step budget (adaptive step is
// capped at 0.7, so crossing 2 * 2000 units needs ~5700 steps).
TraceResult traceFar(const Ray& r, float rEscape) {
  TraceParams p;
  p.rEscape = rEscape;
  p.maxSteps = 20000;
  return trace(r, p);
}

// Bisection on the impact parameter using trace(): Horizon below b_c, Escaped
// above. Writes the midpoint of the final bracket to *result. Every probe must
// be decisive: a MaxSteps outcome (step budget exhausted) is a failure, never
// silently treated as "escaped". Fatal assertions => call through
// ASSERT_NO_FATAL_FAILURE.
void bisectCriticalImpact(float D, double* result) {
  TraceParams p;
  p.rEscape = D + 10.0f;
  p.maxSteps = 4000;
  double lo = 2.0, hi = 3.2;
  // The initial bracket must be valid: lo is captured, hi escapes.
  ASSERT_EQ(trace(impactRay(D, static_cast<float>(lo)), p).outcome,
            Outcome::Horizon)
      << "bracket low end b = " << lo << " must be captured, D = " << D;
  ASSERT_EQ(trace(impactRay(D, static_cast<float>(hi)), p).outcome,
            Outcome::Escaped)
      << "bracket high end b = " << hi << " must escape, D = " << D;
  for (int i = 0; i < 25; ++i) {
    const double mid = 0.5 * (lo + hi);
    const TraceResult t = trace(impactRay(D, static_cast<float>(mid)), p);
    ASSERT_NE(t.outcome, Outcome::MaxSteps)
        << "undecided probe at b = " << mid << ", iteration " << i;
    if (t.outcome == Outcome::Horizon) {
      lo = mid;
    } else {
      ASSERT_EQ(t.outcome, Outcome::Escaped);
      hi = mid;
    }
  }
  *result = 0.5 * (lo + hi);
}

// --- Vec3 -------------------------------------------------------------------

TEST(GeodesicVec3Test, ArithmeticAndProducts) {
  const Vec3 a{1.0f, 2.0f, 3.0f};
  const Vec3 b{-4.0f, 5.0f, 0.5f};

  const Vec3 s = a + b;
  EXPECT_FLOAT_EQ(s.x, -3.0f);
  EXPECT_FLOAT_EQ(s.y, 7.0f);
  EXPECT_FLOAT_EQ(s.z, 3.5f);

  const Vec3 d = a - b;
  EXPECT_FLOAT_EQ(d.x, 5.0f);
  EXPECT_FLOAT_EQ(d.y, -3.0f);
  EXPECT_FLOAT_EQ(d.z, 2.5f);

  const Vec3 m1 = a * 2.0f;
  const Vec3 m2 = 2.0f * a;
  EXPECT_FLOAT_EQ(m1.y, 4.0f);
  EXPECT_FLOAT_EQ(m2.z, 6.0f);

  EXPECT_FLOAT_EQ(dot(a, b), -4.0f + 10.0f + 1.5f);

  // (1,0,0) x (0,1,0) = (0,0,1); right-handed.
  const Vec3 c = cross({1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
  EXPECT_FLOAT_EQ(c.x, 0.0f);
  EXPECT_FLOAT_EQ(c.y, 0.0f);
  EXPECT_FLOAT_EQ(c.z, 1.0f);
  const Vec3 c2 = cross(a, b);
  EXPECT_FLOAT_EQ(c2.x, 2.0f * 0.5f - 3.0f * 5.0f);
  EXPECT_FLOAT_EQ(c2.y, 3.0f * -4.0f - 1.0f * 0.5f);
  EXPECT_FLOAT_EQ(c2.z, 1.0f * 5.0f - 2.0f * -4.0f);
}

TEST(GeodesicVec3Test, LengthAndNormalize) {
  EXPECT_FLOAT_EQ(length({3.0f, 4.0f, 12.0f}), 13.0f);
  const Vec3 n = normalize({0.0f, 3.0f, 4.0f});
  EXPECT_FLOAT_EQ(n.x, 0.0f);
  EXPECT_FLOAT_EQ(n.y, 0.6f);
  EXPECT_FLOAT_EQ(n.z, 0.8f);
  EXPECT_NEAR(length(normalize({-7.0f, 2.0f, 5.0f})), 1.0f, 1e-6f);
}

// --- constants --------------------------------------------------------------

TEST(GeodesicConstantsTest, SchwarzschildLandmarks) {
  EXPECT_FLOAT_EQ(kRs, 1.0f);
  EXPECT_FLOAT_EQ(kPhotonSphere, 1.5f);
  EXPECT_FLOAT_EQ(kIsco, 3.0f);
  EXPECT_NEAR(kCriticalImpact, 1.5 * std::sqrt(3.0), 1e-6);
}

// --- acceleration -----------------------------------------------------------

TEST(GeodesicAccelerationTest, AnalyticValueOnAxis) {
  // a = -1.5 * 3 * (2,0,0) / 2^5 = (-0.28125, 0, 0)
  const Vec3 a = acceleration({2.0f, 0.0f, 0.0f}, 3.0f);
  EXPECT_NEAR(a.x, -0.28125f, 1e-7f);
  EXPECT_NEAR(a.y, 0.0f, 1e-9f);
  EXPECT_NEAR(a.z, 0.0f, 1e-9f);
}

TEST(GeodesicAccelerationTest, AnalyticValueOffAxis) {
  // |pos| = 3, r^5 = 243, h2 = 4: a = -(1.5 * 4 / 243) * (1, 2, 2).
  const double k = -1.5 * 4.0 / 243.0;
  const Vec3 a = acceleration({1.0f, 2.0f, 2.0f}, 4.0f);
  EXPECT_NEAR(a.x, k * 1.0, 1e-8);
  EXPECT_NEAR(a.y, k * 2.0, 1e-8);
  EXPECT_NEAR(a.z, k * 2.0, 1e-8);
}

TEST(GeodesicAccelerationTest, MagnitudeAtPhotonSphereIsCentripetal) {
  // At r = 1.5 with h2 = r^2 (tangential speed 1 at this instant): |a| = 1.5 / r^2 = 2/3
  // = v^2 / r, which is exactly why the photon sphere admits a circular orbit.
  const Vec3 a = acceleration({0.0f, 1.5f, 0.0f}, 2.25f);
  EXPECT_NEAR(length(a), 2.0f / 3.0f, 1e-6f);
}

TEST(GeodesicAccelerationTest, PointsTowardsOrigin) {
  const Vec3 pos{3.0f, -4.0f, 12.0f};  // |pos| = 13
  const Vec3 a = acceleration(pos, 7.0f);
  EXPECT_LT(dot(a, pos), 0.0f);
  // Antiparallel: cross product vanishes, |a . pos| = |a| |pos|.
  EXPECT_NEAR(length(cross(a, pos)), 0.0f, 1e-6f * length(a) * 13.0f);
  EXPECT_NEAR(-dot(a, pos), length(a) * 13.0f, 1e-5f * length(a) * 13.0f);
}

TEST(GeodesicAccelerationTest, ScalesAsInverseFourthPowerOfRadius) {
  // |a| = 1.5 h2 / r^4: doubling r divides by 16 (a Newtonian 1/r^2 law
  // would give 4, an r^-3 law 8).
  const float h2 = 5.0f;
  const float a1 = length(acceleration({2.0f, 0.0f, 0.0f}, h2));
  const float a2 = length(acceleration({4.0f, 0.0f, 0.0f}, h2));
  EXPECT_NEAR(a1 / a2, 16.0f, 1e-4f);
}

TEST(GeodesicAccelerationTest, LinearInH2AndZeroForRadialRay) {
  const Vec3 pos{2.0f, 1.0f, -1.0f};
  const Vec3 a1 = acceleration(pos, 2.0f);
  const Vec3 a2 = acceleration(pos, 4.0f);
  EXPECT_NEAR(a2.x, 2.0f * a1.x, 1e-6f);
  EXPECT_NEAR(a2.y, 2.0f * a1.y, 1e-6f);
  EXPECT_NEAR(a2.z, 2.0f * a1.z, 1e-6f);

  const Vec3 z = acceleration(pos, 0.0f);
  EXPECT_FLOAT_EQ(z.x, 0.0f);
  EXPECT_FLOAT_EQ(z.y, 0.0f);
  EXPECT_FLOAT_EQ(z.z, 0.0f);
}

// --- makeRay ----------------------------------------------------------------

TEST(GeodesicMakeRayTest, NormalizesDirectionAndKeepsPosition) {
  const Ray r = makeRay({-20.0f, 5.0f, 0.0f}, {3.0f, 0.0f, 0.0f});
  EXPECT_FLOAT_EQ(r.pos.x, -20.0f);
  EXPECT_FLOAT_EQ(r.pos.y, 5.0f);
  EXPECT_FLOAT_EQ(r.pos.z, 0.0f);
  EXPECT_NEAR(length(r.vel), 1.0f, 1e-6f);
  EXPECT_NEAR(r.vel.x, 1.0f, 1e-6f);
  EXPECT_NEAR(r.vel.y, 0.0f, 1e-6f);
}

TEST(GeodesicMakeRayTest, H2IsImpactParameterSquared) {
  // Start (-D, b, 0), travel along +x: h = |pos x vel| = b.
  const Ray r = makeRay({-20.0f, 5.0f, 0.0f}, {1.0f, 0.0f, 0.0f});
  EXPECT_NEAR(r.h2, 25.0f, 1e-4f);

  const Ray r2 = makeRay({-100.0f, 0.5f, 0.0f}, {2.0f, 0.0f, 0.0f});
  EXPECT_NEAR(r2.h2, 0.25f, 1e-6f);
}

TEST(GeodesicMakeRayTest, ObliqueRayMatchesDSinAlpha) {
  // b = D sin(alpha) for a ray from distance D aimed at angle alpha off the
  // radial direction.
  const float D = 30.0f, b = 7.0f;
  const Ray r = impactRay(D, b);
  EXPECT_NEAR(length(r.pos), D, 1e-4f);
  EXPECT_NEAR(length(r.vel), 1.0f, 1e-6f);
  EXPECT_NEAR(r.h2, b * b, 1e-3f);
  EXPECT_NEAR(length(cross(r.pos, r.vel)), b, 1e-4f);
}

TEST(GeodesicMakeRayTest, RadialRayHasZeroH2) {
  const Ray r = makeRay({-30.0f, 0.0f, 0.0f}, {5.0f, 0.0f, 0.0f});
  EXPECT_FLOAT_EQ(r.h2, 0.0f);
  const Ray r2 = makeRay({0.0f, 0.0f, 9.0f}, {0.0f, 0.0f, -1.0f});
  EXPECT_FLOAT_EQ(r2.h2, 0.0f);
}

// --- single steps -----------------------------------------------------------

TEST(GeodesicStepperTest, StraightLineWhenH2IsZero) {
  for (int which = 0; which < 2; ++which) {
    Ray r = makeRay({-10.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f});
    for (int i = 0; i < 10; ++i) {
      if (which == 0) {
        stepVerlet(r, 0.5f);
      } else {
        stepRk4(r, 0.5f);
      }
    }
    EXPECT_NEAR(r.pos.x, -5.0f, 1e-5f) << "stepper " << which;
    EXPECT_NEAR(r.pos.y, 0.0f, 1e-6f);
    EXPECT_NEAR(r.vel.x, 1.0f, 1e-6f);
    EXPECT_FLOAT_EQ(r.h2, 0.0f);
  }
}

TEST(GeodesicStepperTest, VerletMatchesVelocityVerletFormula) {
  // x1 = x + v dt + a0 dt^2 / 2;  v1 = v + (a0 + a(x1)) dt / 2.
  // Reference evaluated in double. A correct float step agrees to ~1e-7; a
  // different scheme (Euler, RK4, ...) differs by >= 1e-4 for this step size.
  const double dt = 0.25;
  Ray r = makeRay({3.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});  // h2 = 9
  const D3 x0 = toD(r.pos), v0 = toD(r.vel);
  const D3 a0 = accelD(x0, 9.0);
  const D3 x1 = x0 + v0 * dt + a0 * (0.5 * dt * dt);
  const D3 v1 = v0 + (a0 + accelD(x1, 9.0)) * (0.5 * dt);

  stepVerlet(r, static_cast<float>(dt));
  EXPECT_NEAR(r.pos.x, x1.x, 1e-6);
  EXPECT_NEAR(r.pos.y, x1.y, 1e-6);
  EXPECT_NEAR(r.vel.x, v1.x, 1e-6);
  EXPECT_NEAR(r.vel.y, v1.y, 1e-6);
  EXPECT_EQ(r.h2, 9.0f);
  // The pull is towards the origin: it bends the velocity towards -x.
  EXPECT_LT(r.vel.x, 0.0f);
}

TEST(GeodesicStepperTest, Rk4MatchesClassicRk4) {
  const double dt = 0.25;
  Ray r = makeRay({3.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});  // h2 = 9
  D3 p = toD(r.pos), v = toD(r.vel);
  rk4D(p, v, 9.0, dt);

  stepRk4(r, static_cast<float>(dt));
  EXPECT_NEAR(r.pos.x, p.x, 1e-6);
  EXPECT_NEAR(r.pos.y, p.y, 1e-6);
  EXPECT_NEAR(r.vel.x, v.x, 1e-6);
  EXPECT_NEAR(r.vel.y, v.y, 1e-6);
  EXPECT_EQ(r.h2, 9.0f);
}

// --- conservation laws ------------------------------------------------------

// Steps a strongly bent ray (b = 4, closest approach ~3.5 rs) with the
// adaptive step until it leaves r = 40, checking invariants after every step.
// Returns the number of steps taken.
template <typename Stepper>
int runConservationCheck(Stepper step, double maxRelEnergyDrift) {
  Ray r = impactRay(20.0f, 4.0f);
  const double h0 = std::sqrt(static_cast<double>(r.h2));
  const double e0 = energy(r);
  const float h2Initial = r.h2;
  bool h2Unchanged = true;
  double maxHDrift = 0.0, maxH2Mismatch = 0.0, maxEnergyDrift = 0.0;
  int n = 0;
  for (; n < 2000; ++n) {
    step(r, adaptiveStep(length(r.pos), false));

    // h2 is carried unchanged (exact, not just approximately).
    h2Unchanged = h2Unchanged && (r.h2 == h2Initial);

    // True conserved quantity recomputed from the state: h = |pos x vel|.
    const double h = length(cross(r.pos, r.vel));
    maxHDrift = std::max(maxHDrift, std::fabs(h - h0) / h0);
    // ... and its square must agree with the stored h2 (state consistency).
    maxH2Mismatch = std::max(maxH2Mismatch, std::fabs(h * h - r.h2) / r.h2);
    // Energy |v|^2 - rs h2 / r^3 (E ~ 1 here). Not conserved by straight
    // lines or by speed renormalisation, so this catches both.
    maxEnergyDrift = std::max(maxEnergyDrift, std::fabs(energy(r) - e0));

    if (length(r.pos) >= 40.0f && dot(r.pos, r.vel) > 0.0f) break;
    if (length(r.pos) <= kRs) break;
  }
  EXPECT_TRUE(h2Unchanged);
  // Spec bound: relative drift of h < 1e-3 (a correct float run gives ~1e-6).
  EXPECT_LT(maxHDrift, 1e-3);
  EXPECT_LT(maxH2Mismatch, 2e-3);
  EXPECT_LT(maxEnergyDrift, maxRelEnergyDrift * std::fabs(e0));
  return n;
}

TEST(GeodesicConservationTest, VerletConservesAngularMomentumAndEnergy) {
  // Adaptive dt = 0.12 r is coarse; measured energy drift of a correct
  // velocity-Verlet is 1.6e-3 (relative to E0 ~ 1), so allow 1e-2.
  const int n =
      runConservationCheck([](Ray& r, float dt) { stepVerlet(r, dt); }, 1e-2);
  EXPECT_GT(n, 50);  // the ray really was integrated across the domain
  EXPECT_LT(n, 500);
}

TEST(GeodesicConservationTest, Rk4ConservesAngularMomentumAndEnergy) {
  // RK4 energy drift is ~2e-6 on the same path; allow 1e-4.
  const int n =
      runConservationCheck([](Ray& r, float dt) { stepRk4(r, dt); }, 1e-4);
  EXPECT_GT(n, 50);
  EXPECT_LT(n, 500);
}

TEST(GeodesicConservationTest, TraceReturnsConsistentFinalState) {
  const Ray r0 = impactRay(20.0f, 4.0f);
  const TraceResult t = trace(r0, TraceParams{});
  ASSERT_EQ(t.outcome, Outcome::Escaped);
  EXPECT_EQ(t.ray.h2, r0.h2);
  const double h = length(cross(t.ray.pos, t.ray.vel));
  EXPECT_NEAR(h * h, r0.h2, 2e-3 * r0.h2);
  EXPECT_LT(std::fabs(energy(t.ray) - energy(r0)),
            1e-2 * std::fabs(energy(r0)));
}

// --- weak-field deflection --------------------------------------------------

// Far from the hole (R = 2000 rs) the deflection is the classic
//   alpha = 4M/b + (15 pi / 4) (M/b)^2 + O((M/b)^3),   M = rs/2,
// i.e. 2/b + (15 pi / 16) / b^2 in rs units. For b = 50..200 the first
// correction is 3.0%..0.7% of the leading term and the O((M/b)^3) term is
// <= 0.1%. Starting at finite radius R only drops the tails of the deflection
// integral beyond |s| > ~R along the path: the perpendicular pull there is
// 1.5 b^3 / s^5, whose two tails integrate to 0.75 b^3 / R^4, a relative
// (3/8) (b/R)^4 of the 2/b leading term, i.e. <= 4e-5 for b <= 200 (negligible;
// it is NOT of order b^2/R^2). A correct float implementation measured within
// 0.12% of the two-term formula, so 1% is a safe, still discriminating
// tolerance (a factor 2 error in gravity, or Newtonian-style 1/r^2 force, are
// off by >= 50%).
class GeodesicWeakFieldTest : public ::testing::TestWithParam<float> {};

TEST_P(GeodesicWeakFieldTest, DeflectionMatchesTwoTermExpansion) {
  const float b = GetParam();
  const float D = 2000.0f;
  const Ray r0 = impactRay(D, b);
  const TraceResult t = traceFar(r0, D);
  ASSERT_EQ(t.outcome, Outcome::Escaped);
  EXPECT_GE(length(t.ray.pos), D);

  const double alpha = deflectionAngle(r0, t.ray);
  const double firstOrder = 2.0 / b;
  const double twoTerm = firstOrder + 15.0 * kPi / 16.0 / (b * b);
  EXPECT_NEAR(alpha, twoTerm, 0.01 * twoTerm);
  // Spec check: first-order 4M/b = 2 rs/b to within 5% (3.1% at b = 50).
  EXPECT_NEAR(alpha, firstOrder, 0.05 * firstOrder);
  // Attractive gravity: the ray (initially climbing in +y past the hole on
  // its +y side) is bent towards -y.
  EXPECT_LT(t.ray.vel.y, r0.vel.y);
}

INSTANTIATE_TEST_SUITE_P(ImpactParameters, GeodesicWeakFieldTest,
                         ::testing::Values(50.0f, 100.0f, 200.0f));

TEST(GeodesicWeakFieldScalingTest, DeflectionIsInverselyProportionalToB) {
  const float D = 2000.0f;
  auto alphaFor = [&](float b) {
    const Ray r0 = impactRay(D, b);
    const TraceResult t = traceFar(r0, D);
    EXPECT_EQ(t.outcome, Outcome::Escaped);
    return deflectionAngle(r0, t.ray);
  };
  const double a50 = alphaFor(50.0f);
  const double a100 = alphaFor(100.0f);
  const double a200 = alphaFor(200.0f);
  // Pure 1/b gives 2 and 4. The second-order term shifts the exact ratios to
  // 2.015 and 4.09, so accept +-2.5% / +-6% around the ideal values (an
  // 1/b^2 law would give 4 and 16).
  EXPECT_NEAR(a100 / a200, 2.0, 0.05);
  EXPECT_NEAR(a50 / a200, 4.0, 0.25);
}

TEST(GeodesicWeakFieldScalingTest, VeryLargeImpactParameterBarelyDeflects) {
  const float D = 2000.0f, b = 400.0f;
  const Ray r0 = impactRay(D, b);
  const TraceResult t = traceFar(r0, D);
  EXPECT_EQ(t.outcome, Outcome::Escaped);
  const double alpha = deflectionAngle(r0, t.ray);
  // 2/b = 5e-3; finite start radius costs only (3/8)(b/R)^4 ~ 0.06% here,
  // second order adds +0.4%.
  EXPECT_GT(alpha, 0.0045);
  EXPECT_LT(alpha, 0.0055);
}

// --- strong-field deflection vs independent double reference ----------------

TEST(GeodesicStrongFieldTest, DeflectionMatchesDoublePrecisionReference) {
  // Adaptive-step Verlet (dt up to 0.12 r) is only 2nd order, so allow 0.5%;
  // measured errors are 0.13% (b=4, 6) and 0.02% (b=10). Deflections are
  // 0.86, 0.45 and 0.24 rad; Newtonian-style or factor-2 gravity errors are
  // >> 10% off.
  const float D = 40.0f;
  for (float b : {4.0f, 6.0f, 10.0f}) {
    const Ray r0 = impactRay(D, b);

    D3 p = toD(r0.pos), v = toD(r0.vel);
    for (int n = 0; n < 100000; ++n) {
      rk4D(p, v, r0.h2, 0.004);
      if (n > 100 && dotD(p, p) >= D * D && dotD(p, v) > 0.0) break;
    }
    const double refAlpha =
        std::atan2(std::fabs(v.x * r0.vel.y - v.y * r0.vel.x),
                   v.x * r0.vel.x + v.y * r0.vel.y);

    TraceParams tp;
    tp.rEscape = D;
    tp.maxSteps = 5000;
    const TraceResult t = trace(r0, tp);
    ASSERT_EQ(t.outcome, Outcome::Escaped) << "b = " << b;
    EXPECT_NEAR(deflectionAngle(r0, t.ray), refAlpha, 0.005 * refAlpha)
        << "b = " << b;
  }
}

// --- capture / escape threshold ---------------------------------------------

TEST(GeodesicCaptureTest, JustBelowCriticalImpactIsCaptured) {
  for (float D : {20.0f, 30.0f, 40.0f}) {
    const TraceResult t = trace(impactRay(D, 0.97f * kCriticalImpact),
                                TraceParams{40.0f + 10.0f, 2000, false});
    EXPECT_EQ(t.outcome, Outcome::Horizon) << "D = " << D;
  }
}

TEST(GeodesicCaptureTest, JustAboveCriticalImpactEscapes) {
  for (float D : {20.0f, 30.0f, 40.0f}) {
    const TraceResult t = trace(impactRay(D, 1.03f * kCriticalImpact),
                                TraceParams{40.0f + 10.0f, 2000, false});
    EXPECT_EQ(t.outcome, Outcome::Escaped) << "D = " << D;
  }
}

TEST(GeodesicCaptureTest, RadialRayIsCaptured) {
  const TraceResult t =
      trace(makeRay({-30.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}), TraceParams{});
  EXPECT_EQ(t.outcome, Outcome::Horizon);
}

TEST(GeodesicCaptureTest, CriticalImpactParameterBracketedByBisection) {
  // Exact b_c = 1.5 sqrt(3) rs = 2.5981. The adaptive 2nd-order Verlet
  // integration shifts the measured threshold by about -0.65% (2.581), so a
  // +-1.1% window around the exact value [2.57, 2.63] is satisfiable yet still
  // excludes e.g. a Newtonian-like deflection (b_c ~ 2 or no capture at all)
  // and a factor-2 error in the force (b_c ~ 1.8 or 3.7).
  for (float D : {20.0f, 40.0f}) {
    double bc = 0.0;
    SCOPED_TRACE(::testing::Message() << "D = " << D);
    ASSERT_NO_FATAL_FAILURE(bisectCriticalImpact(D, &bc));
    EXPECT_GE(bc, 2.57) << "D = " << D;
    EXPECT_LE(bc, 2.63) << "D = " << D;
  }
}

// --- circular photon orbit --------------------------------------------------

TEST(GeodesicPhotonSphereTest, CircularOrbitAtOnePointFiveRs) {
  // At r = 1.5 rs with unit tangential velocity, h2 = r^2 and
  // |a| = 1.5 h2 / r^4 = 1/r = v^2 / r: exactly centripetal, so the orbit is
  // circular with angular rate 1/r. It is unstable (Lyapunov rate ~ 0.4 per
  // unit time), but with RK4 at dt = 0.002 the truncation error is ~1e-12 and
  // float noise (~1e-7) is amplified by only e^(0.4 * 9.4) ~ 40 over one full
  // orbit (T = 2 pi r = 9.42), so a correct implementation stays within
  // 3e-4 relative radius (measured); the test allows 1% => >30x headroom,
  // while a 5% error in the force breaks the orbit within a quarter turn.
  const float r0 = kPhotonSphere;
  Ray r = makeRay({r0, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
  ASSERT_NEAR(r.h2, r0 * r0, 1e-5f);

  const float dt = 0.002f;
  const int steps = static_cast<int>(2.0 * kPi * r0 / dt);  // one full orbit
  double maxDev = 0.0;
  for (int i = 1; i <= steps; ++i) {
    stepRk4(r, dt);
    maxDev = std::max(maxDev,
                      static_cast<double>(std::fabs(length(r.pos) - r0) / r0));
    if (i == steps / 4) {
      // A quarter orbit later the photon is at +y (angle advances at 1/r).
      EXPECT_NEAR(r.pos.x, 0.0f, 0.01f);
      EXPECT_NEAR(r.pos.y, r0, 0.01f);
      // Velocity points along -x (counter-clockwise motion).
      EXPECT_NEAR(r.vel.x, -1.0f, 0.01f);
      EXPECT_NEAR(r.vel.y, 0.0f, 0.01f);
    }
  }
  EXPECT_LT(maxDev, 0.01);
  // Back near the start after a full orbit.
  EXPECT_NEAR(r.pos.x, r0, 0.02f);
  EXPECT_NEAR(r.pos.y, 0.0f, 0.02f);
}

// --- order of convergence ---------------------------------------------------

// Float-state error at time T of a scattering ray (b = 3.5 from D = 10,
// closest approach ~3 rs) against a double-precision RK4 reference with
// dt = 1e-3. Float round-off after <= 64 steps is ~1e-6 in position; the
// smallest truncation error used below is 1e-4 (Verlet) / 2.5e-5 (RK4), so
// round-off perturbs the ratios by at most a few percent. Measured ratios:
// Verlet 3.99, 3.96; RK4 14.1, 14.5 -- inside the accepted windows below.
class GeodesicConvergenceTest : public ::testing::Test {
 protected:
  static constexpr double kT = 12.8;  // divisible by every dt used below

  void SetUp() override {
    r0_ = impactRay(10.0f, 3.5f);
    D3 p = toD(r0_.pos), v = toD(r0_.vel);
    const int n = 12800;
    for (int i = 0; i < n; ++i) rk4D(p, v, r0_.h2, kT / n);
    ref_ = p;
  }

  template <typename Stepper>
  double errorAt(Stepper step, double dt) const {
    Ray r = r0_;
    const int n = static_cast<int>(std::lround(kT / dt));
    for (int i = 0; i < n; ++i) step(r, static_cast<float>(dt));
    const D3 d{r.pos.x - ref_.x, r.pos.y - ref_.y, r.pos.z - ref_.z};
    return std::sqrt(dotD(d, d));
  }

  Ray r0_{};
  D3 ref_{};
};

TEST_F(GeodesicConvergenceTest, VerletIsSecondOrder) {
  auto verlet = [](Ray& r, float dt) { stepVerlet(r, dt); };
  const double e04 = errorAt(verlet, 0.4);
  const double e02 = errorAt(verlet, 0.2);
  const double e01 = errorAt(verlet, 0.1);
  EXPECT_GT(e01, 1e-5);              // truncation error dominates round-off
  EXPECT_NEAR(e04 / e02, 4.0, 1.0);  // accept 3..5
  EXPECT_NEAR(e02 / e01, 4.0, 1.0);
  EXPECT_LT(e01, e02);
  EXPECT_LT(e02, e04);
}

TEST_F(GeodesicConvergenceTest, Rk4IsFourthOrder) {
  auto rk4 = [](Ray& r, float dt) { stepRk4(r, dt); };
  const double e16 = errorAt(rk4, 1.6);
  const double e08 = errorAt(rk4, 0.8);
  const double e04 = errorAt(rk4, 0.4);
  EXPECT_GT(e04, 1e-5);
  EXPECT_GT(e16 / e08, 10.0);  // accept 10..20
  EXPECT_LT(e16 / e08, 20.0);
  EXPECT_GT(e08 / e04, 10.0);
  EXPECT_LT(e08 / e04, 20.0);
}

TEST_F(GeodesicConvergenceTest, Rk4IsMoreAccurateThanVerletAtEqualStep) {
  auto verlet = [](Ray& r, float dt) { stepVerlet(r, dt); };
  auto rk4 = [](Ray& r, float dt) { stepRk4(r, dt); };
  EXPECT_LT(errorAt(rk4, 0.4), 0.1 * errorAt(verlet, 0.4));
}

// --- adaptiveStep -----------------------------------------------------------

TEST(GeodesicAdaptiveStepTest, ProportionalToRadiusInsideTheClamp) {
  EXPECT_NEAR(adaptiveStep(1.0f, false), 0.12f, 1e-6f);
  EXPECT_NEAR(adaptiveStep(1.5f, false), 0.18f, 1e-6f);
  EXPECT_NEAR(adaptiveStep(3.0f, false), 0.36f, 1e-6f);
  EXPECT_NEAR(adaptiveStep(5.0f, false), 0.60f, 1e-6f);
}

TEST(GeodesicAdaptiveStepTest, ClampsAtBothEnds) {
  // Upper clamp: 0.12 r >= 0.7 for r >= 5.83.
  EXPECT_FLOAT_EQ(adaptiveStep(10.0f, false), 0.7f);
  EXPECT_FLOAT_EQ(adaptiveStep(2000.0f, false), 0.7f);
  // Lower clamp: 0.12 r <= 0.02 for r <= 1/6 (well inside the horizon).
  EXPECT_FLOAT_EQ(adaptiveStep(0.1f, false), 0.02f);
  EXPECT_FLOAT_EQ(adaptiveStep(0.0f, false), 0.02f);
}

TEST(GeodesicAdaptiveStepTest, NearDiskCapsAt0p085) {
  EXPECT_FLOAT_EQ(adaptiveStep(3.0f, true), 0.085f);    // 0.36 -> 0.085
  EXPECT_FLOAT_EQ(adaptiveStep(10.0f, true), 0.085f);   // 0.7  -> 0.085
  EXPECT_NEAR(adaptiveStep(0.5f, true), 0.06f, 1e-6f);  // below the cap
  EXPECT_FLOAT_EQ(adaptiveStep(0.1f, true), 0.02f);     // lower clamp kept
}

TEST(GeodesicAdaptiveStepTest, MonotoneNonDecreasingInRadius) {
  for (bool nearDisk : {false, true}) {
    float prev = adaptiveStep(0.0f, nearDisk);
    for (float r = 0.05f; r <= 50.0f; r += 0.05f) {
      const float dt = adaptiveStep(r, nearDisk);
      EXPECT_GE(dt, prev) << "r = " << r << " nearDisk = " << nearDisk;
      EXPECT_GE(dt, 0.02f);
      EXPECT_LE(dt, nearDisk ? 0.085f : 0.7f);
      prev = dt;
    }
  }
}

// --- trace bookkeeping ------------------------------------------------------

TEST(GeodesicTraceTest, DefaultParams) {
  const TraceParams p;
  EXPECT_FLOAT_EQ(p.rEscape, 40.0f);
  EXPECT_EQ(p.maxSteps, 2000);
  EXPECT_FALSE(p.nearDisk);
}

TEST(GeodesicTraceTest, HorizonRayEndsInsideHorizon) {
  const Ray r0 = impactRay(30.0f, 0.5f * kCriticalImpact);
  const TraceParams p;
  const TraceResult t = trace(r0, p);
  ASSERT_EQ(t.outcome, Outcome::Horizon);
  EXPECT_GT(t.steps, 0);
  EXPECT_LT(t.steps, p.maxSteps);
  EXPECT_LE(length(t.ray.pos), kRs);
  // The last step is at most 0.12 r ~ 0.15 long, so the ray cannot have
  // tunnelled deep inside the horizon.
  EXPECT_GT(length(t.ray.pos), 0.7f);
  EXPECT_EQ(t.ray.h2, r0.h2);
}

TEST(GeodesicTraceTest, EscapedRayEndsBeyondEscapeRadiusMovingOutward) {
  const Ray r0 = impactRay(30.0f, 2.0f * kCriticalImpact);
  TraceParams p;
  p.rEscape = 40.0f;
  const TraceResult t = trace(r0, p);
  ASSERT_EQ(t.outcome, Outcome::Escaped);
  EXPECT_GT(t.steps, 0);
  EXPECT_LT(t.steps, p.maxSteps);
  EXPECT_GE(length(t.ray.pos), p.rEscape);
  EXPECT_GT(dot(t.ray.pos, t.ray.vel), 0.0f);
  // The step that crosses rEscape overshoots by at most the max step (0.7).
  EXPECT_LT(length(t.ray.pos), p.rEscape + 0.71f);
  EXPECT_EQ(t.ray.h2, r0.h2);
}

TEST(GeodesicTraceTest, MaxStepsExhaustionReportsMaxSteps) {
  const Ray r0 = impactRay(30.0f, 5.0f);
  TraceParams p;
  p.maxSteps = 5;
  const TraceResult t = trace(r0, p);
  EXPECT_EQ(t.outcome, Outcome::MaxSteps);
  EXPECT_EQ(t.steps, 5);
  // The state advanced: 5 steps of 0.7 along a nearly straight line.
  EXPECT_NEAR(length(t.ray.pos - r0.pos), 3.5f, 0.05f);
}

// Largest absolute component difference between two full ray states.
double rayStateDistance(const Ray& a, const Ray& b) {
  return std::max({std::fabs(static_cast<double>(a.pos.x - b.pos.x)),
                   std::fabs(static_cast<double>(a.pos.y - b.pos.y)),
                   std::fabs(static_cast<double>(a.pos.z - b.pos.z)),
                   std::fabs(static_cast<double>(a.vel.x - b.vel.x)),
                   std::fabs(static_cast<double>(a.vel.y - b.vel.y)),
                   std::fabs(static_cast<double>(a.vel.z - b.vel.z)),
                   std::fabs(static_cast<double>(a.h2 - b.h2))});
}

TEST(GeodesicTraceTest, SingleStepIsOneVerletStepOfAdaptiveSize) {
  // trace() must advance with stepVerlet (not stepRk4) using
  // dt = adaptiveStep(|pos| at the start of the step, nearDisk). Start at
  // r = 4 (strongly curved, h2 = 14.4 => b ~ 3.8, nonlinear motion) where the
  // far-field step is dt = 0.48, big enough that Verlet and RK4 land visibly
  // apart (measured ~4.7e-4 in position, 1.4e-4 in velocity, vs ~1e-7 float
  // noise).
  const Ray r0 = makeRay({4.0f, 0.0f, 0.0f}, {-0.3f, 0.9f, 0.0f});
  ASSERT_GT(r0.h2, 10.0f);
  TraceParams p;
  p.maxSteps = 1;
  p.nearDisk = false;

  const float dt = adaptiveStep(length(r0.pos), p.nearDisk);
  ASSERT_NEAR(dt, 0.48f, 1e-6f);  // 0.12 * 4, inside the clamp

  Ray verlet = r0;
  stepVerlet(verlet, dt);
  Ray rk4 = r0;
  stepRk4(rk4, dt);

  const TraceResult t = trace(r0, p);
  EXPECT_EQ(t.outcome, Outcome::MaxSteps);
  EXPECT_EQ(t.steps, 1);
  // Entire state (pos, vel, h2) equals one Verlet step; tolerance is a few
  // float ulps at |component| <= 4 (same operations, so ~0 in practice).
  EXPECT_LT(rayStateDistance(t.ray, verlet), 1e-6);
  EXPECT_EQ(t.ray.h2, r0.h2);

  // Discrimination: the RK4 step must be measurably different at this
  // configuration, >= 100x the tolerance above, so an RK4-based trace fails.
  EXPECT_GT(rayStateDistance(rk4, verlet), 1e-4);
  EXPECT_GT(rayStateDistance(t.ray, rk4), 1e-4);

  // The same holds with the small near-disk step (equality with Verlet only;
  // RK4 and Verlet agree too closely at dt = 0.085 to discriminate).
  p.nearDisk = true;
  const float dtNear = adaptiveStep(length(r0.pos), true);
  ASSERT_FLOAT_EQ(dtNear, 0.085f);
  Ray verletNear = r0;
  stepVerlet(verletNear, dtNear);
  EXPECT_LT(rayStateDistance(trace(r0, p).ray, verletNear), 1e-6);
}

TEST(GeodesicTraceTest, NearDiskUsesSmallSteps) {
  // A radial ray feels no force (h2 = 0) so it moves exactly dt per step along
  // +x: 10 steps of 0.085 (nearDisk) vs 10 steps of 0.7 (far field).
  const Ray r0 = makeRay({-30.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f});
  TraceParams p;
  p.maxSteps = 10;

  p.nearDisk = true;
  const TraceResult near = trace(r0, p);
  EXPECT_EQ(near.outcome, Outcome::MaxSteps);
  EXPECT_NEAR(near.ray.pos.x, -30.0f + 10 * 0.085f, 1e-4f);

  p.nearDisk = false;
  const TraceResult far = trace(r0, p);
  EXPECT_EQ(far.outcome, Outcome::MaxSteps);
  EXPECT_NEAR(far.ray.pos.x, -30.0f + 10 * 0.7f, 1e-4f);
}

TEST(GeodesicTraceTest, EscapeRequiresOutwardMotion) {
  // Starting beyond rEscape but moving inward must NOT count as escaped.
  TraceParams p;
  p.rEscape = 40.0f;
  p.maxSteps = 3;
  const TraceResult in =
      trace(makeRay({-50.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}), p);
  EXPECT_EQ(in.outcome, Outcome::MaxSteps);
  EXPECT_EQ(in.steps, 3);

  // Beyond rEscape and moving outward: escapes on the very first step.
  const TraceResult out =
      trace(makeRay({-50.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}), p);
  EXPECT_EQ(out.outcome, Outcome::Escaped);
  EXPECT_EQ(out.steps, 1);
  EXPECT_NEAR(length(out.ray.pos), 50.7f, 1e-3f);
}

TEST(GeodesicTraceTest, RayStartingInsideHorizonIsCapturedImmediately) {
  const TraceResult t =
      trace(makeRay({0.0f, 0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}), TraceParams{});
  EXPECT_EQ(t.outcome, Outcome::Horizon);
  EXPECT_EQ(t.steps, 1);
}

// --- nearDiskPlane (step-size policy near the accretion disk) ---------------
//
// adaptiveStep's 0.085 cap is only needed where the integrator can cross the
// disk plane inside the disk: |y| < 1 and r <= diskOuter + 1. Everywhere else
// the normal step policy (dt up to 0.7) must apply, otherwise the maxSteps
// budget (4000 * 0.085 = 340 units of path) runs out for far observers.

TEST(NearDiskPlaneTest, TrueNearThePlaneInsideTheDisk) {
  EXPECT_TRUE(nearDiskPlane({5.0f, 0.0f, 0.0f}, 12.0f));
  EXPECT_TRUE(nearDiskPlane({0.0f, 0.0f, -8.0f}, 12.0f));
  EXPECT_TRUE(nearDiskPlane({3.0f, 0.5f, 4.0f}, 12.0f));
  EXPECT_TRUE(nearDiskPlane({3.0f, 0.99f, 4.0f}, 12.0f));
  EXPECT_TRUE(nearDiskPlane({3.0f, -0.99f, 4.0f}, 12.0f));  // |y|
  EXPECT_TRUE(nearDiskPlane({0.0f, 0.0f, 0.0f}, 12.0f));
}

TEST(NearDiskPlaneTest, FalseAwayFromThePlane) {
  EXPECT_FALSE(nearDiskPlane({5.0f, 1.01f, 0.0f}, 12.0f));
  EXPECT_FALSE(nearDiskPlane({5.0f, -1.01f, 0.0f}, 12.0f));
  EXPECT_FALSE(nearDiskPlane({0.0f, 5.0f, 0.0f}, 12.0f));
  EXPECT_FALSE(nearDiskPlane({1.0f, 3.0f, 1.0f}, 12.0f));
  EXPECT_FALSE(nearDiskPlane({0.0f, 20.0f, -20.0f}, 12.0f));  // the camera
}

TEST(NearDiskPlaneTest, FalseBeyondTheOuterRadiusPlusOne) {
  EXPECT_TRUE(nearDiskPlane({12.5f, 0.0f, 0.0f}, 12.0f));
  EXPECT_TRUE(nearDiskPlane({0.0f, 0.0f, 13.0f}, 12.0f));   // r == outer + 1
  EXPECT_FALSE(nearDiskPlane({13.1f, 0.0f, 0.0f}, 12.0f));
  EXPECT_FALSE(nearDiskPlane({0.0f, 0.0f, -20.0f}, 12.0f));
  EXPECT_FALSE(nearDiskPlane({100.0f, 0.0f, 0.0f}, 12.0f));
  EXPECT_FALSE(nearDiskPlane({0.0f, 0.5f, -150.0f}, 12.0f));
  // r is the 3-D distance, not the in-plane one.
  EXPECT_FALSE(nearDiskPlane({9.0f, 0.9f, 9.5f}, 12.0f));  // r = 13.1
}

TEST(NearDiskPlaneTest, ThresholdScalesWithDiskOuter) {
  const Vec3 p{20.0f, 0.0f, 0.0f};
  EXPECT_FALSE(nearDiskPlane(p, 12.0f));
  EXPECT_FALSE(nearDiskPlane(p, 18.0f));
  EXPECT_TRUE(nearDiskPlane(p, 19.0f));
  EXPECT_TRUE(nearDiskPlane(p, 30.0f));
  EXPECT_TRUE(nearDiskPlane({3.0f, 0.0f, 0.0f}, 3.5f));
  EXPECT_FALSE(nearDiskPlane({5.0f, 0.0f, 0.0f}, 3.5f));
}

}  // namespace
