// CPU-only tests for the accretion-disk math (include/blackhole/disk.hpp).
//
// Units: rs = 1, c = 1, M = 0.5. All expected values are hand-derived. Useful
// closed form used throughout: with v = sqrt(0.5 / (r - 1)),
//   sqrt(1 - 1/r) / gamma(v) = sqrt((r - 1.5) / r),
// so g(r, c) = sqrt((r - 1.5) / r) / (1 - v c).
// Float tolerances: a handful of float operations => ~1e-6 relative error, so
// 1e-5 (scalars) / 1e-4 (chains with pow) are safe yet far below the gaps
// between the correct value and each plausible mutant.

#include "blackhole/disk.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace disk = blackhole::disk;
using disk::Crossing;
using disk::DiskParams;
using disk::Vec3;

namespace {

constexpr float kSqrt2 = 1.41421356f;

float maxChannel(Vec3 c) { return std::max({c.x, c.y, c.z}); }

float vOf(float r) { return std::sqrt(0.5f / (r - 1.0f)); }

// Independent closed form of the Doppler/gravitational factor.
float gRef(float r, float c) {
  return std::sqrt((r - 1.5f) / r) / (1.0f - vOf(r) * c);
}

void expectVec(Vec3 v, float x, float y, float z, float tol = 1e-6f) {
  EXPECT_NEAR(v.x, x, tol);
  EXPECT_NEAR(v.y, y, tol);
  EXPECT_NEAR(v.z, z, tol);
}

}  // namespace

// ---------------------------------------------------------------- Vec3

TEST(DiskVec3Test, AddSubScale) {
  const Vec3 a{1, 2, 3};
  const Vec3 b{4, -5, 6};
  expectVec(a + b, 5, -3, 9);
  expectVec(a - b, -3, 7, -3);
  expectVec(a * 2.0f, 2, 4, 6);
  expectVec(2.0f * a, 2, 4, 6);
}

TEST(DiskVec3Test, DotCrossLengthNormalize) {
  EXPECT_FLOAT_EQ(disk::dot({1, 2, 3}, {4, -5, 6}), 12.0f);
  expectVec(disk::cross({1, 0, 0}, {0, 1, 0}), 0, 0, 1);
  expectVec(disk::cross({0, 1, 0}, {1, 0, 0}), 0, 0, -1);
  expectVec(disk::cross({1, 2, 3}, {4, 5, 6}), -3, 6, -3);
  EXPECT_FLOAT_EQ(disk::length({2, 3, 6}), 7.0f);
  expectVec(disk::normalize({2, 3, 6}), 2.0f / 7, 3.0f / 7, 6.0f / 7);
  expectVec(disk::normalize({0, 0, 5}), 0, 0, 1);
}

// ---------------------------------------------------------------- params

TEST(DiskParamsTest, Defaults) {
  const DiskParams p;
  EXPECT_FLOAT_EQ(p.rInner, 3.0f);
  EXPECT_FLOAT_EQ(p.rOuter, 12.0f);
  EXPECT_FLOAT_EQ(p.tInnerK, 12000.0f);
  EXPECT_FLOAT_EQ(p.brightness, 1.0f);
}

TEST(DiskParamsTest, InDiskBoundariesInclusive) {
  const DiskParams p;
  EXPECT_TRUE(disk::inDisk(3.0f, p));
  EXPECT_TRUE(disk::inDisk(12.0f, p));
  EXPECT_TRUE(disk::inDisk(7.5f, p));
  EXPECT_FALSE(disk::inDisk(std::nextafter(3.0f, 0.0f), p));
  EXPECT_FALSE(disk::inDisk(std::nextafter(12.0f, 100.0f), p));
  EXPECT_FALSE(disk::inDisk(0.0f, p));
  EXPECT_FALSE(disk::inDisk(-5.0f, p));
  EXPECT_FALSE(disk::inDisk(100.0f, p));
  EXPECT_FALSE(disk::inDisk(std::numeric_limits<float>::quiet_NaN(), p));
}

TEST(DiskParamsTest, InDiskHonoursCustomRadii) {
  DiskParams p;
  p.rInner = 2.0f;
  p.rOuter = 5.0f;
  EXPECT_TRUE(disk::inDisk(2.0f, p));
  EXPECT_TRUE(disk::inDisk(5.0f, p));
  EXPECT_FALSE(disk::inDisk(1.99f, p));
  EXPECT_FALSE(disk::inDisk(5.01f, p));
  EXPECT_FALSE(disk::inDisk(8.0f, p));
}

// ---------------------------------------------------------------- T, L

TEST(DiskTemperatureTest, PowerLawMinusThreeQuarters) {
  const DiskParams p;
  EXPECT_NEAR(disk::diskTemperature(3.0f, p), 12000.0f, 12000.0f * 1e-5f);
  // 12000 * 2^-0.75 = 7135.5; an exponent of -1/2 would give 8485.
  EXPECT_NEAR(disk::diskTemperature(6.0f, p), 12000.0f * std::pow(2.0f, -0.75f),
              7136.0f * 1e-4f);
  EXPECT_NEAR(disk::diskTemperature(6.0f, p), 7135.5f, 1.0f);
  // 12000 * 4^-0.75 = 12000 / (2 sqrt 2).
  EXPECT_NEAR(disk::diskTemperature(12.0f, p), 4242.64f, 0.5f);
}

TEST(DiskTemperatureTest, UsesParamsAndDecreasesOutward) {
  DiskParams p;
  p.rInner = 2.0f;
  p.tInnerK = 5000.0f;
  EXPECT_NEAR(disk::diskTemperature(2.0f, p), 5000.0f, 0.1f);
  // 5000 * 4^-0.75 = 1767.77
  EXPECT_NEAR(disk::diskTemperature(8.0f, p), 1767.77f, 0.2f);
  float prev = disk::diskTemperature(2.0f, p);
  for (float r = 2.5f; r <= 12.0f; r += 0.5f) {
    const float t = disk::diskTemperature(r, p);
    EXPECT_LT(t, prev) << r;
    prev = t;
  }
}

TEST(DiskTemperatureTest, RadiantBrightnessIsStefanBoltzmann) {
  const DiskParams p;
  EXPECT_NEAR(disk::radiantBrightness(12000.0f, p), 1.0f, 1e-6f);
  EXPECT_NEAR(disk::radiantBrightness(6000.0f, p), 1.0f / 16.0f, 1e-6f);
  EXPECT_NEAR(disk::radiantBrightness(24000.0f, p), 16.0f, 1e-4f);
  EXPECT_FLOAT_EQ(disk::radiantBrightness(0.0f, p), 0.0f);
  // T(r=6)^4 / T_in^4 = 2^-3 (T^4 ~ r^-3).
  EXPECT_NEAR(disk::radiantBrightness(disk::diskTemperature(6.0f, p), p),
              0.125f, 0.125f * 1e-4f);
  DiskParams q;
  q.brightness = 2.5f;
  EXPECT_NEAR(disk::radiantBrightness(disk::diskTemperature(6.0f, q), q),
              2.5f * 0.125f, 2.5f * 0.125f * 1e-4f);
  q.tInnerK = 6000.0f;
  EXPECT_NEAR(disk::radiantBrightness(6000.0f, q), 2.5f, 1e-5f);
}

// ---------------------------------------------------------------- orbit

TEST(DiskOrbitTest, LocalSpeedValues) {
  EXPECT_NEAR(disk::orbitalSpeedLocal(3.0f), 0.5f, 1e-6f);
  EXPECT_NEAR(disk::orbitalSpeedLocal(9.0f), 0.25f, 1e-6f);
  EXPECT_NEAR(disk::orbitalSpeedLocal(5.0f), 0.35355339f, 1e-6f);
  EXPECT_NEAR(disk::orbitalSpeedLocal(2.0f), 0.70710678f, 1e-6f);
  EXPECT_NEAR(disk::orbitalSpeedLocal(1.5f), 1.0f, 1e-6f);  // photon sphere
}

TEST(DiskOrbitTest, DirectionIsCounterClockwiseFromPlusY) {
  expectVec(disk::orbitalDirection({3, 0, 0}), 0, 0, -1);
  expectVec(disk::orbitalDirection({0, 0, 3}), 1, 0, 0);
  expectVec(disk::orbitalDirection({-3, 0, 0}), 0, 0, 1);
  expectVec(disk::orbitalDirection({0, 0, -3}), -1, 0, 0);
}

TEST(DiskOrbitTest, DirectionUnitPerpendicularIgnoresY) {
  const Vec3 pts[] = {{3, 0, 0}, {2, 0, -4}, {-7.5f, 0, 1.25f}, {0.3f, 0, 9}};
  for (const Vec3& p : pts) {
    const Vec3 d = disk::orbitalDirection(p);
    EXPECT_NEAR(std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z), 1.0f, 1e-6f);
    EXPECT_NEAR(d.x * p.x + d.y * p.y + d.z * p.z, 0.0f, 1e-5f);
    EXPECT_NEAR(d.y, 0.0f, 1e-7f);
    const Vec3 up{p.x, 5.0f, p.z};
    const Vec3 dn{p.x, -2.0f, p.z};
    expectVec(disk::orbitalDirection(up), d.x, d.y, d.z, 1e-6f);
    expectVec(disk::orbitalDirection(dn), d.x, d.y, d.z, 1e-6f);
  }
}

// ---------------------------------------------------------------- g

TEST(DiskDopplerTest, GFactorAtIsco) {
  EXPECT_NEAR(disk::gFactor(3.0f, 1.0f), kSqrt2, 1e-5f);
  EXPECT_NEAR(disk::gFactor(3.0f, -1.0f), 0.4714045f, 1e-5f);
  EXPECT_NEAR(disk::gFactor(3.0f, 0.0f), 0.7071068f, 1e-5f);
  EXPECT_NEAR(disk::gFactor(3.0f, 0.5f), 0.7071068f / 0.75f, 1e-5f);
}

TEST(DiskDopplerTest, GFactorAtR9) {
  EXPECT_NEAR(disk::gFactor(9.0f, 1.0f), 1.2171612f, 1e-5f);
  EXPECT_NEAR(disk::gFactor(9.0f, -1.0f), 0.7302967f, 1e-5f);
  EXPECT_NEAR(disk::gFactor(9.0f, 0.0f), 0.9128709f, 1e-5f);
}

TEST(DiskDopplerTest, MatchesClosedFormAndIsMonotoneInCos) {
  for (float r : {2.0f, 3.0f, 4.5f, 6.0f, 12.0f, 1000.0f}) {
    float prev = 0.0f;
    for (float c = -1.0f; c <= 1.0f; c += 0.25f) {
      const float g = disk::gFactor(r, c);
      EXPECT_NEAR(g, gRef(r, c), 1e-5f * gRef(r, c)) << r << " " << c;
      EXPECT_GT(g, prev) << r << " " << c;
      prev = g;
    }
  }
  // Far away: no Doppler, no gravity => g -> 1.
  EXPECT_NEAR(disk::gFactor(1.0e6f, 0.3f), 1.0f, 1e-3f);
}

TEST(DiskDopplerTest, ApproachingOverRecedingIntensityRatio) {
  // I ~ g^3 => ((1+v)/(1-v))^3; at ISCO v = 1/2 => 27 exactly.
  const float ratio = std::pow(disk::gFactor(3.0f, 1.0f), 3.0f) /
                      std::pow(disk::gFactor(3.0f, -1.0f), 3.0f);
  EXPECT_NEAR(ratio, 27.0f, 27.0f * 1e-4f);
  // r = 9, v = 1/4: (1.25/0.75)^3 = 4.62963.
  const float ratio9 = std::pow(disk::gFactor(9.0f, 1.0f), 3.0f) /
                       std::pow(disk::gFactor(9.0f, -1.0f), 3.0f);
  EXPECT_NEAR(ratio9, 4.62963f, 4.62963f * 1e-4f);
}

TEST(DiskDopplerTest, IntensityBoostIsCubic) {
  EXPECT_NEAR(disk::intensityBoost(1.0f), 1.0f, 1e-6f);
  EXPECT_NEAR(disk::intensityBoost(2.0f), 8.0f, 1e-5f);
  EXPECT_NEAR(disk::intensityBoost(0.5f), 0.125f, 1e-6f);
  EXPECT_NEAR(disk::intensityBoost(kSqrt2), 2.828427f, 1e-4f);
  EXPECT_FLOAT_EQ(disk::intensityBoost(0.0f), 0.0f);
}

// Ray travels camera -> disk. Photon heads to the observer along -rayDir.
// cosTheta = dot(orbitalDirection(pos), -rayDir).
TEST(DiskDopplerTest, GFactorForRayGasApproachingObserver) {
  // Gas at (3,0,0) moves along -z. Camera at -z looks along +z (ray +z); the
  // photon goes -z, the same way as the gas: blueshift, cosTheta = +1.
  EXPECT_NEAR(disk::gFactorForRay({3, 0, 0}, {0, 0, 1}), kSqrt2, 1e-5f);
  // Gas at (0,0,9) moves along +x. Ray along -x => photon along +x.
  EXPECT_NEAR(disk::gFactorForRay({0, 0, 9}, {-1, 0, 0}), 1.2171612f, 1e-5f);
}

TEST(DiskDopplerTest, GFactorForRayGasRecedingFromObserver) {
  // Gas at (-3,0,0) moves along +z; ray +z => photon -z: cosTheta = -1.
  EXPECT_NEAR(disk::gFactorForRay({-3, 0, 0}, {0, 0, 1}), 0.4714045f, 1e-5f);
  // Gas at (3,0,0) moves along -z; camera at +z looking back (ray -z) =>
  // photon goes +z: receding.
  EXPECT_NEAR(disk::gFactorForRay({3, 0, 0}, {0, 0, -1}), 0.4714045f, 1e-5f);
}

TEST(DiskDopplerTest, GFactorForRayTransverseAndOblique) {
  EXPECT_NEAR(disk::gFactorForRay({3, 0, 0}, {0, -1, 0}), 0.7071068f, 1e-5f);
  const float s = std::sqrt(0.5f);
  // -ray = (0,s,-s); orbital dir (0,0,-1) => cos = s = 0.70711.
  EXPECT_NEAR(disk::gFactorForRay({3, 0, 0}, {0, -s, s}),
              0.7071068f / (1.0f - 0.5f * s), 1e-5f);
}

TEST(DiskDopplerTest, GFactorForRayGeneralPosition) {
  const Vec3 pos{2, 0, -4};
  const float r = std::sqrt(20.0f);
  const float inv = 1.0f / std::sqrt(1.0f + 4.0f + 9.0f);
  const Vec3 ray{1 * inv, -2 * inv, 3 * inv};
  // orbital dir = (z, 0, -x)/r = (-4, 0, -2)/r
  const float cosT = (-4 * -ray.x + -2 * -ray.z) / r;
  EXPECT_NEAR(disk::gFactorForRay(pos, ray), gRef(r, cosT),
              1e-5f * gRef(r, cosT));
}

// ---------------------------------------------------------------- blackbody

TEST(DiskBlackbodyTest, NormalisedAndEndpoints) {
  for (float t = 1000.0f; t <= 40000.0f; t *= 1.03f) {
    const Vec3 c = disk::blackbodyColor(t);
    EXPECT_TRUE(std::isfinite(c.x) && std::isfinite(c.y) && std::isfinite(c.z));
    EXPECT_GE(std::min({c.x, c.y, c.z}), 0.0f) << t;
    EXPECT_NEAR(maxChannel(c), 1.0f, 1e-6f) << t;
  }
  EXPECT_NEAR(disk::blackbodyColor(1000.0f).x, 1.0f, 1e-6f);
  EXPECT_NEAR(disk::blackbodyColor(40000.0f).z, 1.0f, 1e-6f);
}

TEST(DiskBlackbodyTest, ClampsOutsideRange) {
  const Vec3 lo = disk::blackbodyColor(1000.0f);
  const Vec3 hi = disk::blackbodyColor(40000.0f);
  for (float t : {500.0f, 1.0f, 0.0f, -300.0f}) {
    const Vec3 c = disk::blackbodyColor(t);
    expectVec(c, lo.x, lo.y, lo.z, 1e-6f);
  }
  for (float t : {50000.0f, 1.0e6f}) {
    const Vec3 c = disk::blackbodyColor(t);
    expectVec(c, hi.x, hi.y, hi.z, 1e-6f);
  }
}

TEST(DiskBlackbodyTest, ColdIsOrange) {
  for (float t : {1000.0f, 1500.0f, 2000.0f, 2500.0f, 3000.0f}) {
    const Vec3 c = disk::blackbodyColor(t);
    EXPECT_GT(c.x, c.y) << t;
    EXPECT_GT(c.y, c.z) << t;
  }
}

TEST(DiskBlackbodyTest, MidIsNearWhite) {
  const Vec3 c = disk::blackbodyColor(6500.0f);
  EXPECT_GE(std::min({c.x, c.y, c.z}), 0.85f * maxChannel(c));
}

TEST(DiskBlackbodyTest, HotIsBlue) {
  for (float t : {20000.0f, 25000.0f, 30000.0f, 40000.0f}) {
    const Vec3 c = disk::blackbodyColor(t);
    EXPECT_GE(c.z, c.y) << t;
    EXPECT_GE(c.y, c.x) << t;
    EXPECT_NEAR(c.z, 1.0f, 1e-6f) << t;
  }
}

// Anchors from CIE 1931 integration of Planck's law (Wyman-Sloan-Shirley
// fits), linear sRGB, max channel = 1. +-0.08 absolute leaves room for LUT /
// fit approximations but rejects gamma-encoded output (3000 K: g would be
// ~0.7 instead of 0.48) and wrong colour spaces.
TEST(DiskBlackbodyTest, MatchesPlanckianAnchors) {
  struct A {
    float t, r, g, b;
  };
  const A anchors[] = {{2000, 1.0f, 0.266f, 0.008f},
                       {3000, 1.0f, 0.484f, 0.155f},
                       {6500, 1.0f, 0.945f, 0.993f},
                       {10000, 0.610f, 0.695f, 1.0f},
                       {20000, 0.407f, 0.536f, 1.0f}};
  for (const A& a : anchors) {
    const Vec3 c = disk::blackbodyColor(a.t);
    EXPECT_NEAR(c.x, a.r, 0.08f) << a.t;
    EXPECT_NEAR(c.y, a.g, 0.08f) << a.t;
    EXPECT_NEAR(c.z, a.b, 0.08f) << a.t;
  }
}

TEST(DiskBlackbodyTest, BlueOverRedMonotoneAndColoursContinuous) {
  Vec3 prev = disk::blackbodyColor(1000.0f);
  float prevRatio = prev.z / prev.x;
  for (float t = 1000.0f * 1.01f; t <= 40000.0f; t *= 1.01f) {
    const Vec3 c = disk::blackbodyColor(t);
    const float ratio = c.z / c.x;
    EXPECT_GE(ratio, prevRatio) << t;  // non-decreasing everywhere
    if (t >= 2500.0f) {
      EXPECT_GT(ratio, prevRatio) << t;  // strictly increasing from 2500 K
    }
    EXPECT_LT(std::fabs(c.x - prev.x), 0.05f) << t;
    EXPECT_LT(std::fabs(c.y - prev.y), 0.05f) << t;
    EXPECT_LT(std::fabs(c.z - prev.z), 0.05f) << t;
    prev = c;
    prevRatio = ratio;
  }
}

// ---------------------------------------------------------------- radiance

TEST(DiskRadianceTest, ZeroOutsideDiskAndNoNaNOnDegenerateRadii) {
  const DiskParams p;
  for (float r : {-1.0f, 0.0f, 0.5f, 1.0f, 1.5f, 2.99f, 12.01f, 100.0f}) {
    expectVec(disk::diskRadiance(r, 0.3f, p), 0, 0, 0, 0.0f);
  }
}

TEST(DiskRadianceTest, FiniteNonNegativeInsideDisk) {
  const DiskParams p;
  for (float r = 3.0f; r <= 12.0f; r += 0.75f) {
    for (float c = -1.0f; c <= 1.0f; c += 0.25f) {
      const Vec3 v = disk::diskRadiance(r, c, p);
      EXPECT_TRUE(std::isfinite(v.x) && std::isfinite(v.y) &&
                  std::isfinite(v.z))
          << r << " " << c;
      EXPECT_GE(std::min({v.x, v.y, v.z}), 0.0f);
      EXPECT_GT(maxChannel(v), 0.0f);
    }
  }
}

// blackbodyColor has max channel 1, so max(radiance) = L(T) * g^3 exactly.
TEST(DiskRadianceTest, PeakChannelIsBrightnessTimesGCubed) {
  const DiskParams p;
  EXPECT_NEAR(maxChannel(disk::diskRadiance(3.0f, 1.0f, p)), 2.828427f,
              2.83e-4f);
  EXPECT_NEAR(maxChannel(disk::diskRadiance(3.0f, -1.0f, p)), 0.104757f,
              1.1e-5f);
  // r = 6, c = 0: L = 1/8, g = sqrt(0.75) => g^3 = 0.649519 => 0.0811899.
  EXPECT_NEAR(maxChannel(disk::diskRadiance(6.0f, 0.0f, p)), 0.0811899f, 1e-5f);
  for (float r : {3.0f, 4.5f, 6.0f, 9.0f, 12.0f}) {
    for (float c : {-1.0f, -0.4f, 0.0f, 0.6f, 1.0f}) {
      const float expected =
          std::pow(r / 3.0f, -3.0f) * std::pow(gRef(r, c), 3.0f);
      EXPECT_NEAR(maxChannel(disk::diskRadiance(r, c, p)), expected,
                  expected * 2e-4f)
          << r << " " << c;
    }
  }
}

TEST(DiskRadianceTest, ColourIsBlackbodyAtObservedTemperature) {
  const DiskParams p;
  for (float r : {3.0f, 6.0f, 12.0f}) {
    for (float c : {-1.0f, 0.0f, 1.0f}) {
      const float g = gRef(r, c);
      const float tObs = g * disk::diskTemperature(r, p);
      const Vec3 bb = disk::blackbodyColor(tObs);
      const float scale = maxChannel(disk::diskRadiance(r, c, p));
      const Vec3 v = disk::diskRadiance(r, c, p);
      EXPECT_NEAR(v.x / scale, bb.x, 2e-3f) << r << " " << c;
      EXPECT_NEAR(v.y / scale, bb.y, 2e-3f) << r << " " << c;
      EXPECT_NEAR(v.z / scale, bb.z, 2e-3f) << r << " " << c;
    }
  }
}

TEST(DiskRadianceTest, ApproachingSideBrighterAndBluer) {
  const DiskParams p;
  for (float r : {3.0f, 9.0f}) {
    const Vec3 a = disk::diskRadiance(r, 1.0f, p);
    const Vec3 b = disk::diskRadiance(r, -1.0f, p);
    const float v = vOf(r);
    const float expected = std::pow((1 + v) / (1 - v), 3.0f);
    EXPECT_NEAR(maxChannel(a) / maxChannel(b), expected, expected * 2e-4f) << r;
    EXPECT_GT(a.z / a.x, b.z / b.x) << r;
  }
}

TEST(DiskRadianceTest, DecreasesOutwardAtFixedCosTheta) {
  const DiskParams p;
  for (float c : {-0.5f, 0.0f, 0.5f}) {
    float prev = maxChannel(disk::diskRadiance(3.0f, c, p));
    for (float r = 4.0f; r <= 12.0f; r += 1.0f) {
      const float cur = maxChannel(disk::diskRadiance(r, c, p));
      EXPECT_LT(cur, prev) << r << " " << c;
      prev = cur;
    }
  }
}

TEST(DiskRadianceTest, HonoursBrightnessAndInnerTemperature) {
  DiskParams p;
  p.brightness = 2.0f;
  p.tInnerK = 6000.0f;
  const Vec3 v = disk::diskRadiance(3.0f, 0.0f, p);
  EXPECT_NEAR(maxChannel(v), 2.0f * std::pow(0.7071068f, 3.0f), 1e-4f);
  // Observed temperature 0.7071 * 6000 = 4243 K, not 8485 K (12000 default).
  const Vec3 bb = disk::blackbodyColor(0.7071068f * 6000.0f);
  EXPECT_NEAR(v.z / maxChannel(v), bb.z, 2e-3f);
  EXPECT_NEAR(v.y / maxChannel(v), bb.y, 2e-3f);
}

// ---------------------------------------------------------------- crossing

TEST(DiskPlaneCrossingTest, HitsWithInterpolation) {
  Crossing c = disk::planeCrossing({0, 1, 0}, {2, -1, 4});
  ASSERT_TRUE(c.hit);
  EXPECT_NEAR(c.t, 0.5f, 1e-6f);
  expectVec(c.pos, 1, 0, 2);
  EXPECT_EQ(c.pos.y, 0.0f);

  c = disk::planeCrossing({2, -1, 4}, {0, 1, 0});
  ASSERT_TRUE(c.hit);
  EXPECT_NEAR(c.t, 0.5f, 1e-6f);
  expectVec(c.pos, 1, 0, 2);

  c = disk::planeCrossing({0, 3, 0}, {4, -1, 8});
  ASSERT_TRUE(c.hit);
  EXPECT_NEAR(c.t, 0.75f, 1e-6f);
  expectVec(c.pos, 3, 0, 6);
}

TEST(DiskPlaneCrossingTest, PosYExactlyZeroForInexactInterpolation) {
  const Vec3 starts[] = {{1, 0.1f, 2}, {0, 0.7f, -3}, {5, 0.3f, 5}};
  const Vec3 ends[] = {{2, -0.3f, 1}, {1, -0.1f, 2}, {4, -0.9f, 7}};
  for (const Vec3& a : starts) {
    for (const Vec3& b : ends) {
      const Crossing c = disk::planeCrossing(a, b);
      ASSERT_TRUE(c.hit);
      EXPECT_EQ(c.pos.y, 0.0f);
      EXPECT_GE(c.t, 0.0f);
      EXPECT_LE(c.t, 1.0f);
      EXPECT_NEAR(c.t, a.y / (a.y - b.y), 1e-6f);
      EXPECT_NEAR(c.pos.x, a.x + c.t * (b.x - a.x), 1e-5f);
      EXPECT_NEAR(c.pos.z, a.z + c.t * (b.z - a.z), 1e-5f);
    }
  }
  // Pairs for which the naive float lerp of y leaves a residue (~1e-8).
  const float pairs[][2] = {{0.1f, -1.4f}, {0.1f, -2.4f}, {0.2f, -2.8f}};
  for (const auto& yy : pairs) {
    const Crossing c = disk::planeCrossing({1, yy[0], 2}, {3, yy[1], 4});
    ASSERT_TRUE(c.hit);
    EXPECT_EQ(c.pos.y, 0.0f) << yy[0] << " " << yy[1];
  }
  const Crossing tiny = disk::planeCrossing({0, 1e-30f, 0}, {2, -1e-30f, 2});
  ASSERT_TRUE(tiny.hit);
  EXPECT_NEAR(tiny.t, 0.5f, 1e-6f);
}

TEST(DiskPlaneCrossingTest, MissesWhenBothSameSide) {
  EXPECT_FALSE(disk::planeCrossing({0, 1, 0}, {1, 2, 1}).hit);
  EXPECT_FALSE(disk::planeCrossing({0, -1, 0}, {1, -2, 1}).hit);
  EXPECT_FALSE(disk::planeCrossing({0, 0.001f, 0}, {1, 5, 1}).hit);
}

TEST(DiskPlaneCrossingTest, EndingExactlyOnPlaneIsAHitAtT1) {
  Crossing c = disk::planeCrossing({1, 2, 3}, {5, 0, 7});
  ASSERT_TRUE(c.hit);
  EXPECT_NEAR(c.t, 1.0f, 1e-6f);
  expectVec(c.pos, 5, 0, 7);
  c = disk::planeCrossing({1, -2, 3}, {5, 0, 7});
  ASSERT_TRUE(c.hit);
  EXPECT_NEAR(c.t, 1.0f, 1e-6f);
  expectVec(c.pos, 5, 0, 7);
}

TEST(DiskPlaneCrossingTest, StartingOnPlaneIsNotAHit) {
  EXPECT_FALSE(disk::planeCrossing({1, 0, 3}, {5, 2, 7}).hit);
  EXPECT_FALSE(disk::planeCrossing({1, 0, 3}, {5, -2, 7}).hit);
  EXPECT_FALSE(disk::planeCrossing({1, 0, 3}, {5, 0, 7}).hit);
}

TEST(DiskPlaneCrossingTest, VertexOnPlaneCountsExactlyOnce) {
  const float ys[][3] = {{2, 0, -2}, {-2, 0, 2}, {2, 0, 2}, {-2, 0, -2}};
  const bool expectedHits[] = {true, true, true, true};
  for (int i = 0; i < 4; ++i) {
    const int h = disk::planeCrossing({0, ys[i][0], 0}, {0, ys[i][1], 0}).hit +
                  disk::planeCrossing({0, ys[i][1], 0}, {0, ys[i][2], 0}).hit;
    EXPECT_EQ(h, expectedHits[i] ? 1 : 0) << i;
  }
}

TEST(DiskPlaneCrossingTest, MirrorSymmetry) {
  const Vec3 as[] = {{0, 1, 0}, {3, 0.3f, -2}, {-1, 5, 4}};
  const Vec3 bs[] = {{2, -1, 4}, {1, -0.7f, 2}, {0, -0.01f, 0}};
  for (const Vec3& a : as) {
    for (const Vec3& b : bs) {
      const Crossing f = disk::planeCrossing(a, b);
      const Crossing r = disk::planeCrossing(b, a);
      ASSERT_TRUE(f.hit);
      ASSERT_TRUE(r.hit);
      EXPECT_NEAR(f.t, 1.0f - r.t, 1e-6f);
      EXPECT_NEAR(f.pos.x, r.pos.x, 1e-5f);
      EXPECT_NEAR(f.pos.z, r.pos.z, 1e-5f);
    }
  }
}
